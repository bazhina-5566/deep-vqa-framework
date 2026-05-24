# Cloud GPU Platform Setup Guide

This guide provides step-by-step instructions for renting and configuring a cloud GPU instance (e.g., AutoDL, Lambda Labs, RunPod) to run the `Deep-VQA-Framework`.

## Choosing the Right Instance

For optimal training performance, select an instance that meets the following criteria:

| Component | Minimum Specification | Recommended for VQA |
| :--- | :--- | :--- |
| **GPU** | **RTX 3060 / 4060** | **RTX 3090 / 4090** |
| **VRAM** | **24GB** | **24GB - 80GB** |
| **Disk** | **100GB SSD** | **200GB+ SSD (Dataset caching)** |

> [!TIP]
> **AutoDL Users**: Select the "PyTorch 2.x + CUDA 12.x + Python 3.10+" base image to save setup time.

---

## Rental Strategy: Cost Optimization

When renting cloud GPU instances, choose the billing method that best fits your task duration to maximize cost-efficiency:

On-Demand (Pay-as-you-go): Best suited for short-term tasks (1–3 hours) such as debugging, smoke tests, or code verification. Remember to terminate the instance immediately after your task is complete to avoid unnecessary costs.

Prepaid (Daily/Weekly): If you are planning a long-running training job (24+ hours), always choose a daily or weekly subscription. This typically offers a 30%–50% discount compared to on-demand rates.

Billing Pitfall: When using on-demand billing, it is easy to forget about active instances. We recommend adding a reminder at the end of your Makefile training pipeline or setting a hard timeout on the cloud platform dashboard.

---

## Environment & Data Storage Recommendations

Cloud platform system disks (Root Disk) often have limited capacity and lower throughput. To ensure peak performance, always store your code and datasets on the high-speed data disk.

Data Disk Mapping: Using AutoDL as an example, ensure that your repository and datasets are located in the /root/autodl-tmp/ directory. This partition is mounted on a high-speed data drive, providing significantly better I/O performance than the system disk.

```bash
# Example for AutoDL: Navigate to the data disk and clone the project
cd /root/autodl-tmp/
git clone https://github.com/autentisitet/deep-vqa-framework.git
```

---

## ⚠️ Important Notes

* **Performance**: Placing the repo cloned in data disk is critical for video training tasks where the I/O bottleneck is often the primary cause of slow training speeds.

* **Automation**: If you frequently use the same cloud provider, you can modify your setup_env.sh script to automatically check if you are in the correct directory and warn you if you are running on the system disk.

* **Billing**: Always check your cloud provider's console to ensure your instance is stopped/terminated after training to avoid unexpected charges.

* **Security**: If you use public cloud instances, change the default SSH password and use SSH keys.

* **Decord Backend**: As mentioned in the main README, if your cloud instance lacks OpenCV video codecs, please ensure `decord` is installed.

---

*If you encounter any platform-specific issues, please refer to the cloud provider's official documentation or open an issue in the repository.*
