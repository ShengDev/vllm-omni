# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Adapted from Physis-AI WaveServe; see provenance.json.
"""One-sided CUDA IPC push with producer-ready and consumer-release flags.

Adapted from mini-wave/paged_kv.py. Mailboxes have bounded fixed addresses;
the manager's owned/replica pages have independent lifetimes. Only setup uses
metadata collectives. Payload rounds use driver copies and device-side waits.
"""

import socket

import torch
import torch.distributed as dist

from .kv_transport import fragments


class IPCPush:
    def __init__(self, policy, device, group=None):
        from cuda.bindings import driver

        self.driver, self.policy = driver, policy
        self.device, self.group = torch.device(device), group
        self.rank, self.world = dist.get_rank(group), dist.get_world_size(group)
        self.stream = torch.cuda.Stream(device=self.device)
        self.opened, self.peers = {}, {}
        self.ticket, self.last, self.closed = 0, {}, False
        self.buffers = self.flags = None
        error, info = None, None
        try:
            if self.device.type != "cuda":
                raise ValueError("IPC KV transport requires CUDA")
            self.buffers = torch.empty(
                (self.world - 1, policy.slots, policy.slot_bytes), dtype=torch.uint8, device=self.device
            )
            # ready[source, generation], released[destination, generation]
            self.flags = torch.zeros((2, self.world, policy.slots), dtype=torch.int32, device=self.device)
            info = (
                socket.gethostname(),
                str(torch.cuda.get_device_properties(self.device).uuid),
                policy,
                self.export(self.buffers) if self.world > 1 else None,
                self.export(self.flags),
            )
        except Exception as exc:
            error = f"rank {self.rank}: {type(exc).__name__}: {exc}"
        peers = self.gather((info, error))
        errors = [error for _, error in peers if error]
        if errors:
            raise RuntimeError("IPC KV allocation/export failed: " + str(errors))
        infos = [info for info, _ in peers]
        if len({i[0] for i in infos}) != 1 or len({i[1] for i in infos}) != self.world:
            raise ValueError("IPC KV requires distinct CUDA devices on one host")
        if any(i[2] != policy for i in infos):
            raise ValueError("ranks disagree on KVTransport")
        error = None
        try:
            for rank, (_, _, _, buffer, flags) in enumerate(infos):
                if rank != self.rank:
                    self.peers[rank] = (self.open(buffer), self.open(flags))
        except Exception as exc:
            error = f"rank {self.rank}: {type(exc).__name__}: {exc}"
        errors = [e for e in self.gather(error) if e]
        if errors:
            for pointer in self.opened.values():
                self.check(driver.cuIpcCloseMemHandle(pointer))
            self.opened.clear()
            self.peers.clear()
            dist.barrier(group=group, device_ids=[self.device.index])
            raise RuntimeError("IPC KV peer mapping failed: " + str(errors))
        torch.cuda.current_stream(self.device).synchronize()
        dist.barrier(group=group, device_ids=[self.device.index])

    def gather(self, value):
        values = [None] * self.world
        dist.all_gather_object(values, value, group=self.group)
        return values

    @staticmethod
    def check(result):
        error, *values = result
        if int(error):
            raise RuntimeError(f"CUDA driver error: {error}")
        return values[0] if len(values) == 1 else values

    def export(self, tensor):
        pointer = tensor.data_ptr()
        handle = self.check(self.driver.cuIpcGetMemHandle(pointer))
        base, _ = self.check(self.driver.cuMemGetAddressRange(pointer))
        return bytes(handle.reserved), pointer - int(base)

    def open(self, descriptor):
        handle_bytes, offset = descriptor
        if handle_bytes not in self.opened:
            handle = self.driver.CUipcMemHandle()
            handle.reserved = handle_bytes
            flag = int(self.driver.CUipcMem_flags.CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS.value)
            self.opened[handle_bytes] = int(self.check(self.driver.cuIpcOpenMemHandle(handle, flag)))
        return self.opened[handle_bytes] + offset

    def copy(self, dst, src, size, stream):
        self.check(self.driver.cuMemcpyDtoDAsync(dst, src, size, stream))

    def write(self, stream, address, value):
        self.check(self.driver.cuStreamWriteValue32(stream, address, value, 0))

    def wait(self, stream, address, value):
        flag = int(self.driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ.value)
        self.check(self.driver.cuStreamWaitValue32(stream, address, value, flag))

    def flag_address(self, base, kind, peer, generation):
        return base + ((kind * self.world + peer) * self.policy.slots + generation) * 4

    def slot_address(self, base, owner, source, generation):
        index = source - int(source > owner)  # No self mailbox.
        return base + (index * self.policy.slots + generation) * self.policy.slot_bytes

    def transfer(self, transfers):
        if self.closed:
            raise RuntimeError("IPC KV transport is closed")
        result, outgoing = {}, {}
        plan = fragments(transfers, self.policy.slot_bytes)
        for index, (identity, src, dst, signature, values) in enumerate(transfers):
            if src == dst:
                if self.rank == dst:
                    result[identity] = values
                continue
            if self.rank == src:
                outgoing[index] = tuple(None if t is None else t.contiguous() for t in values)
            if self.rank == dst:
                result[identity] = tuple(
                    None if shape is None else torch.empty(shape, dtype=dtype, device=self.device)
                    for shape, dtype in signature
                )
        if not plan:
            return result
        current = torch.cuda.current_stream(self.device)
        ready = torch.cuda.Event()
        ready.record(current)
        self.stream.wait_event(ready)
        copy_stream, read_stream = self.stream.cuda_stream, current.cuda_stream
        for round_index in range(max(map(len, plan.values()))):
            self.ticket += 1
            if self.ticket >= 2**31:
                raise RuntimeError("IPC generation counter exhausted; start a new request")
            generation = (self.ticket - 1) % self.policy.slots
            # Enqueue ALL sends before waits, including bidirectional edges.
            for (src, dst), rounds in plan.items():
                if round_index >= len(rounds):
                    continue
                edge = (src, dst, generation)
                if self.rank == src:
                    previous = self.last.get(edge, 0)
                    if previous:
                        self.wait(copy_stream, self.flag_address(self.flags.data_ptr(), 1, dst, generation), previous)
                    buffer, flags = self.peers[dst]
                    target = self.slot_address(buffer, dst, src, generation)
                    for index, field, offset, slot_offset, size in rounds[round_index]:
                        self.copy(target + slot_offset, outgoing[index][field].data_ptr() + offset, size, copy_stream)
                    self.write(copy_stream, self.flag_address(flags, 0, src, generation), self.ticket)
                self.last[edge] = self.ticket
            for (src, dst), rounds in plan.items():
                if self.rank != dst or round_index >= len(rounds):
                    continue
                self.wait(read_stream, self.flag_address(self.flags.data_ptr(), 0, src, generation), self.ticket)
                source = self.slot_address(self.buffers.data_ptr(), dst, src, generation)
                for index, field, offset, slot_offset, size in rounds[round_index]:
                    target = result[transfers[index][0]][field]
                    self.copy(target.data_ptr() + offset, source + slot_offset, size, read_stream)
                # A producer may reuse this mailbox only after these reads finish.
                self.write(read_stream, self.flag_address(self.peers[src][1], 1, dst, generation), self.ticket)
        for values in outgoing.values():
            for tensor in values:
                if tensor is not None:
                    tensor.record_stream(self.stream)
        return result

    def drain(self):
        done = torch.cuda.Event()
        done.record(self.stream)
        torch.cuda.current_stream(self.device).wait_event(done)

    def close(self):
        if self.closed:
            return
        self.drain()
        torch.cuda.current_stream(self.device).synchronize()
        dist.barrier(group=self.group, device_ids=[self.device.index])
        for pointer in self.opened.values():
            self.check(self.driver.cuIpcCloseMemHandle(pointer))
        self.opened.clear()
        self.peers.clear()
        # Exporters must remain alive until EVERY importer has closed its maps.
        dist.barrier(group=self.group, device_ids=[self.device.index])
        self.buffers = self.flags = None
        self.closed = True
