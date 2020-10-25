# Secloccore Protocol

This is a tiny server-client based secure network connection tool.

## secfow

## selocserver


The server listen a port and wait 

## selecclient

1. Connect to Server
2. Handshake
3. Listen ports and wait another app to connect
4. 






## HandShake protocol

- E(K1): encrypted message using key 1
- E(K2): encrypted message using key 2
- remote code: 1 connect ok; 0 connect fail

```
1 Client | ----- E(K1) randomC + HelloCode + randomC ---> | Server
2 Client | <---- E(K1) randomC + HelloCode + randomC ---- | Server
3 Client | ----- E(K1) ChangeKey + randC1 --------------> | Server
4 Client | <---- E(K1) random key K2 + randC2 ----------- | Server
5 Client | ----- E(K2) randC2 remoteaddr ---------------> | Server
6 Client | <---- E(K2) randC1 remote code---------------- | Server
if remote code is 0; close connection
if remote code is 1; create pipe
7 Client | <-- E(K2) message --> | Server | <- message -> | Remote
```

If remoteaddr is empty 

```
...
5 Client | ----- E(K2) randC2 --------------------------> | Server
6 Client | <---- E(K2) randC1 --------------------------- | Server
Close
```



