# sbts-boot-from-SSD
One command installation of boot from SSD for Jetson nano, Xavier NX and Xavier AGX

Simply clone this repo. Change into the clone and run:

sudo ./sbts_install_boot_from_SSD.sh

The system will wipe the SSD that you choose, create a partition the full size of the disk, copy
the SD disk to the new partion and change the boot sequence.

Note:

This approach is quite different from the most other approaches but has the advantage that you can afterwards easily run in a resilient read-only memory OS overlayFS afterwards,similar to this: http://wiki.psuter.ch/doku.php?id=solve_raspbian_sd_card_corruption_issues_with_read-only_mounted_root_partition except with the SSD as the lower layer. Please note, if you wish to do this you need to disable docker.service and docker.socket. If you need docker, then it's possible to still do this by using the sbts-base project instead that creates additional partitions and then migrating the docker data directories to the read-write partition.
