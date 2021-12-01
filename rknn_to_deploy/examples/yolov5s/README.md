# Aarch64 Linux Demo
## build

modify `GCC_COMPILER` on `build-linux.sh` for target platform, then execute

```
./build-linux.sh
```

## install

connect device and push build output into `/userdata`

```
adb push install/rknn_ssd_demo_Linux /userdata/
```

## run

```
adb shell
cd /userdata/rknn_ssd_demo_Linux/
```

- rk3566/rk3568
```
export LD_LIBRARY_PATH=./lib
./rknn_ssd_demo model/ssd_inception_v2.rknn model/road.bmp
```


# Android Demo
## build

modify `ANDROID_NDK_PATH` on `build-android.sh` for target platform, then execute

```
./build-android.sh
```

## install

connect device and push build output into `/data`

```
adb push install/rknn_ssd_demo_Android /data/
```

## run

```
adb shell
cd /data/rknn_ssd_demo_Android/
```

- rk3566/rk3568
```
export LD_LIBRARY_PATH=./lib
./rknn_ssd_demo model/ssd_inception_v2.rknn model/road.bmp
```