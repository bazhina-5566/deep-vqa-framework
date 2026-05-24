# ⚠️ Disclaimer

## 1. Data Integrity & Usage
- **Datasets**: This framework provides scripts to automate the download of public datasets (TID2013, KoNViD-1k, etc.). The author **does not host** these datasets and is not responsible for their availability, integrity, or compliance with the original licenses provided by their respective creators.
- **User Responsibility**: Please ensure that your use of these datasets complies with the Terms of Service of the hosting platforms (e.g., Google Drive, university servers) and the licensing agreements specified by the dataset authors.

## 2. Dependency Management
- **Download Tools**: This framework encourages the use of high-performance downloaders like `aria2` and `gdown`. The author is not responsible for any network security issues, data corruption, or IP rate-limiting caused by the use of third-party download tools or external mirror sites.
- **Environment**: Using `uv` or `venv` is recommended to isolate dependencies. The author is not liable for any system-level conflicts, configuration errors, or hardware damage occurring on cloud instances (e.g., AutoDL) or local workstations during installation or training.

## 3. Resource Consumption
- **Cloud Usage**: Training deep learning models is resource-intensive. The user is responsible for monitoring their cloud GPU usage (e.g., AutoDL balance/billing). The author is not liable for any financial costs incurred during the use of this framework.
- **Hardware Safety**: Training can generate significant heat and stress on GPUs. Please ensure your hardware/cooling system is adequate for long-term high-load operations.

---
