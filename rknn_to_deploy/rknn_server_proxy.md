## 连板调试简介
RKNN Toolkit2的连板功能一般需要更新板端的 rknn_server 和 librknnrt.so，并且手动启动 rknn_server 才能正常工作。  
rknn_server: 是一个运行在板子上的后台代理服务，用于接收PC通过USB传输过来的协议，然后执行板端runtime对应的接口，并返回结果给PC。  
librknnrt.so: 是一个板端的runtime库。

有些固件默认已经集成了rknn_server，如果已经集成，可以忽略下面的启动步骤。


## rknn_server存放目录
### Android平台
```
Android
└── rknn_server
    ├── arm64-v8a
    │   └── vendor
    │       └── bin
    │           └── rknn_server
    └── armeabi-v7a
        └── vendor
            └── bin
                └── rknn_server
```

### Linux平台
```
Linux
└── rknn_server
    ├── aarch64
    │   └── usr
    │       └── bin
    │           ├── restart_rknn.sh
    │           ├── rknn_server
    │           └── start_rknn.sh
    └── armhf
        └── usr
            └── bin
                ├── restart_rknn.sh
                ├── rknn_server
                └── start_rknn.sh
```

## 启动步骤
### Android平台
1. adb root && adb remount
2. adb push Android/rknn_server/${BOARD_ARCH}/rknn_server到板子/vendor/bin/目录
3. adb push Android/librknn_api/${BOARD_ARCH}/librknnrt.so到/vendor/lib64（64位系统特有）和/vendor/lib目录
4. 进入板子的串口终端，执行：
```
su
chmod +x /vendor/bin/rknn_server
sync
reboot
```
5. 重新进入板子的串口终端，执行 `pgrep rknn_server`, 查看是否有 rknn_server的进程id（较新的固件开机会自动启动rknn_server），如果不存在，则手动执行: 
```
su
setenforce 0
/vendor/bin/rknn_server &
```   

### Linux平台
1. adb push Linux/rknn_server/${BOARD_ARCH}/usr/bin/下的所有文件到/usr/bin目录
2. adb push Linux/librknn_api/${BOARD_ARCH}/librknnrt.so到/usr/lib目录
3. 进入板子的串口终端，执行：
```
chmod +x /usr/bin/rknn_server
chmod +x /usr/bin/start_rknn.sh
chmod +x /usr/bin/restart_rknn.sh
restart_rknn.sh
```


### 串口查看rknn_server详细日志
#### Android平台
1. 进入串口终端,设置日志等级
```
su
setenforce 0
export RKNN_SERVER_LOGLEVEL=5
```
2. 杀死rknn_server进程
```
kill -9 `pgrep rknn_server`
```
3. 重启rknn_server进程(若固件没有自启动rknn_server)
```
/vendor/bin/rknn_server &
logcat
```
#### Linux平台
1. 进入串口终端,设置日志等级
```
export RKNN_SERVER_LOGLEVEL=5
```
2. 重启rknn_server进程(若固件没有自启动rknn_server)
```
restart_rknn.sh
```
