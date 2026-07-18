from pathlib import Path

from src.datasets.base_dataset import BaseDataset


class ASVspoofDataset(BaseDataset):
    def __init__(
        self,
        audio_dir,
        protocol_path,
        instance_transforms=None,
        *args,
        **kwargs,
    ):
        audio_dir = Path(audio_dir)
        protocol_path = Path(protocol_path)

        index = self._create_index(audio_dir, protocol_path)

        super().__init__(
            index=index,
            instance_transforms=instance_transforms,
            *args,
            **kwargs,
        )

    @staticmethod
    def _create_index(audio_dir, protocol_path):
        index = []

        label_map = {
            "spoof": 0,
            "bonafide": 1,
        }

        with protocol_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                values = line.strip().split()
                if not values:
                    continue
                record = {
                    "speaker_id": values[0],
                    "audio_file_name": values[1],
                    "system_id": values[3],
                    "key": values[4],
                    "path": audio_dir / f"{values[1]}.flac",
                    "label": label_map[values[4]],
                }
                index.append(record)

        return index
