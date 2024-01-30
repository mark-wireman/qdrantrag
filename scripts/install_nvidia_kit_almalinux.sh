sudo yum install kernel-devel-$(uname -r) kernel-headers-$(uname -r)

sudo dnf install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm

subscription-manager repos --enable=rhel-9-for-x86_64-appstream-rpms
subscription-manager repos --enable=rhel-9-for-x86_64-baseos-rpms
subscription-manager repos --enable=codeready-builder-for-rhel-9-x86_64-rpms

sudo rpm --erase gpg-pubkey-7fa2af80*

sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo

sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo

sudo dnf clean expire-cache

sudo dnf module install nvidia-driver:latest-dkms
sudo dnf install cuda-toolkit

sudo dnf install nvidia-gds

sudo reboot
