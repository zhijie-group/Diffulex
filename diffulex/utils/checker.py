def CHECK_SLOT_MAPPING(seqs, slot_mapping):
    # check slot mapping layout
    start_idx = 0
    for seq in seqs:
        cur_ref_slot_mapping = []
        for idx in range(seq.num_diffusion_blocks):
            if seq.active_blocks[idx]:
                padding_num_tokens = (seq.num_diffusion_blocks - idx) * seq.diffusion_block_size
                cur_ref_slot_mapping.extend([-1] * padding_num_tokens)
                break
            elif seq.to_cache_blocks[idx]:
                cur_ref_slot_mapping.extend([0] * seq.diffusion_block_size)
        cur_slot_mapping = slot_mapping[start_idx:start_idx + len(cur_ref_slot_mapping)]
        for slot, ref_slot in zip(cur_slot_mapping, cur_ref_slot_mapping):
            try:
                if ref_slot == -1:
                    assert slot == -1
                elif ref_slot == 0:
                    assert slot != -1
                elif ref_slot is not None:
                    assert slot is not None
            except AssertionError:
                raise ValueError(f"Slot mapping mismatch: {slot} != {ref_slot}. "
                                    f"Check the implementation of prepare_decode.\n"
                                    f"slot_mapping: {cur_slot_mapping}\n"
                                    f"ref_slot_mapping: {cur_ref_slot_mapping}\n"
                                    f"diff: {[s - r for s, r in zip(cur_slot_mapping, cur_ref_slot_mapping)]}")
        start_idx += len(cur_ref_slot_mapping)