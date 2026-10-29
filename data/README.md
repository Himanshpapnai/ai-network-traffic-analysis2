# Dataset Information

This project uses the **CIC-IDS2017 (Canadian Institute for Cybersecurity Intrusion Detection System)** dataset.

## Dataset Files
The dataset consists of network traffic captures including:
- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`
- `Friday-WorkingHours-Morning.pcap_ISCX.csv`
- `Monday-WorkingHours.pcap_ISCX.csv`
- `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`
- `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`
- `Tuesday-WorkingHours.pcap_ISCX.csv`
- `Wednesday-workingHours.pcap_ISCX.csv`

## Notice on GitHub File Limits
Due to GitHub's **100 MB per-file limit**, raw CSV captures are excluded from this Git repository to prevent push errors and excessive repository bloat.

## Download Instructions
You can obtain the complete CIC-IDS2017 dataset from:
- [UNB CIC-IDS2017 Official Page](https://www.unb.ca/cic/datasets/ids-2017.html)
- [Kaggle CIC-IDS2017 Mirror](https://www.kaggle.com/datasets/cicdataset/cicids2017)

Place raw CSV files into `data/raw/` before running model training:
```bash
python src/train.py
```
