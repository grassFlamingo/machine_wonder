import asyncio
import urllib.parse
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format=
    '%(asctime)s %(filename)s %(levelname)s: %(message)s')


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
               **kwargs):
    try:
        while not reader.at_eof():
            writer.write(await reader.read(1024))
    except Exception as e:
        logging.error(e)


# ref: https://tools.ietf.org/html/draft-luotonen-web-proxy-tunneling-01
async def handle_ssl(client_reader: asyncio.StreamReader,
                             client_writer: asyncio.StreamWriter, hostport,
                             **kwargs):
    logging.info(f"[start] -> {hostport}")
    remote_reader, remote_writer = await asyncio.open_connection(*hostport)

    resp = f"{kwargs['httpversion']} 200 Connection established\r\nProxy-agent: Netscape-Proxy/1.1\r\n\r\n"
    client_writer.write(resp.encode())

    try:
        up = pipe(remote_reader, client_writer)
        down = pipe(client_reader, remote_writer)
        await asyncio.gather(up, down)
    except Exception as e:
        logging.error(e)

    remote_writer.close()
    logging.info(f"[end] <- {hostport}")


async def handle_http(client_reader: asyncio.StreamReader,
                              client_writer: asyncio.StreamWriter, hostport,
                              **kwargs):
    logging.info(f"[start] -> {hostport}")
    remote_reader, remote_writer = await asyncio.open_connection(*hostport)
    remote_writer.write(kwargs['headraw'])

    try:
        up = pipe(remote_reader, client_writer)
        down = pipe(client_reader, remote_writer)
        await asyncio.gather(up, down)
    except Exception as e:
        logging.error(e)

    remote_writer.close()
    logging.info(f"[end] <- {hostport}")


async def handle_queries(client_reader: asyncio.StreamReader,
                         client_writer: asyncio.StreamWriter):
    header_raw = await client_reader.read(1024)
    if not header_raw:
        return
    header = header_raw.decode().split("\r\n")
    hellomsg = header[0].split()

    headerdict = dict()
    for l in header[1:-1]:
        ti = l.find(":")
        headerdict[l[0:ti]] = l[ti + 1::]
    headerdict['headraw'] = header_raw

    if hellomsg[0] == "CONNECT":
        # https connection
        thp = hellomsg[1].split(":")
        outhost = thp[0]
        outport = int(thp[1])
        headerdict["httpversion"] = hellomsg[2]
        await handle_ssl(client_reader, client_writer,
                                 (outhost, outport), **headerdict)
    elif hellomsg[0] == "GET":
        # http connection
        # GET http://www.baidu.com/ HTTP/1.1
        parans = urllib.parse.urlparse(hellomsg[1])
        outhost = parans.hostname
        outport = 80 if parans.port is None else parans.port
        await handle_http(client_reader, client_writer,
                                  (outhost, outport), **headerdict)

    client_writer.close()


HOST, PORT = "0.0.0.0", 7788
loop = asyncio.get_event_loop()
server = loop.run_until_complete(
    asyncio.start_server(handle_queries, HOST, PORT))
logging.info(f"[system] running on {HOST}:{PORT}")

try:
    loop.run_forever()
except Exception as e:
    logging.error(e)
    logging.info("[system] exit")
    server.close()
    loop.close()

