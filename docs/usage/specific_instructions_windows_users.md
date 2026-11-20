# Specific instructions for Windows users


## Move Docker Desktop storage

By default, Docker Desktop stores its data on the C: drive. Users with limited disk space may want to move this data to another drive with more available storage.

1. Open **Docker Desktop → Settings → Resources → Advanced → Disk image location**.
2. Select a folder on the destination drive with enough free space for your Docker data.
3. Apply the changes and restart Docker Desktop if prompted.
4. Verify that your images and containers are available before resuming work.

Use Docker Desktop’s settings to move its disk image [1].

## Move the Ubuntu WSL distribution

This optional procedure exports Ubuntu, removes its current registration, and imports it on another drive. The examples use `D:\WSL`; replace this path with your preferred location. Run Windows commands in **PowerShell**, and Linux commands inside **Ubuntu**.

### Export Ubuntu

Save your work and close running containers and Ubuntu sessions. In PowerShell, create the destination folders, shut down WSL, and export Ubuntu:

```powershell
New-Item -ItemType Directory -Force -Path D:\WSL, D:\WSL\Ubuntu
wsl --shutdown
wsl --export Ubuntu D:\WSL\ubuntu.tar
```
Continue only after the export succeeds and important data is backed up. An archive’s existence alone does not prove that it is a valid backup.

### Import Ubuntu on the destination drive

In PowerShell, remove the existing registration and import the exported distribution:

```powershell
wsl --unregister Ubuntu
wsl --import Ubuntu D:\WSL\Ubuntu D:\WSL\ubuntu.tar --version 2
wsl --set-default Ubuntu
```

**Important:** `wsl --unregister Ubuntu` deletes the registered Ubuntu distribution and its filesystem [2]. 

Keep the exported archive until you have verified the relocated installation and your files.

### Restore your default Linux user

Imported distributions somtimes start as `root` by default [3]. Restore your existing regular Linux user before running `docker.sh`, whose container initialization expects a non-root host UID.

```powershell
ubuntu.exe config --default-user USERNAME
```

### Reconfigure Docker after relocation

Ensure Docker Desktop is running and **Settings → Resources → WSL Integration → Ubuntu** is enabled. You can then return to the [Docker setup instructions](../../Readme.md#docker).

## References

[1] Docker. [Docker Desktop WSL 2 backend on Windows](https://docs.docker.com/desktop/features/wsl/).

[2] Microsoft. [Basic commands for WSL](https://learn.microsoft.com/en-us/windows/wsl/basic-commands).

[3] Microsoft. [Import any Linux distribution to use with WSL](https://learn.microsoft.com/en-us/windows/wsl/use-custom-distro#add-wsl-specific-components-like-a-default-user).
