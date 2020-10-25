# secfow -- Secure Forward

Begin:
```
Client | ---- Some unsafe protocol ---> | Server
```

After:
```
Client | --- unsafe protocol ---> | secfow | --- safe protocol --> | Server
```

