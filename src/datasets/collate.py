import torch


def repeat_pad(audio, target_length):
    """
    Repeat audio while it length less than needed
    """
    channels, current_length = audio.shape
    repeat_count = (target_length + current_length - 1) // current_length
    repeated_audio = audio.repeat(1, repeat_count)
    return repeated_audio[:, :target_length]


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    target_length = 1724 + 128 * (600 - 1)

    audio = [repeat_pad(elem["audio"], target_length) for elem in dataset_items]

    batch = {
        "audio": torch.stack(audio, dim=0),
        "labels": torch.tensor(
            [elem["labels"] for elem in dataset_items],
            dtype=torch.long,
        ),
    }
    for metadata_key in (
        "speaker_id",
        "audio_file_name",
        "system_id",
        "key",
        "path",
    ):
        if metadata_key in dataset_items[0]:
            batch[metadata_key] = [item[metadata_key] for item in dataset_items]

    return batch
