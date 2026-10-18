"""Local TCP peers only: no phone, aircraft, Android build or network discovery."""
import socket
import threading
import time
import unittest
from follow_neo.telemetry import HeadingReceiver
from follow_neo.control_transport import ControlChannel


class HeadingStreamTests(unittest.TestCase):
    def run_peer(self,reject_listen=False,push=True):
        listener=socket.socket();listener.bind(('127.0.0.1',0));listener.listen(1)
        requests=[];events=[];clients=[];done=threading.Event()
        def serve():
            peer=None
            try:
                peer,_=listener.accept();peer.settimeout(.04)
                buffer=b'';subscribed=False
                while not done.is_set():
                    try:data=peer.recv(4096)
                    except socket.timeout:data=None
                    if data==b'':break
                    if data:buffer+=data
                    while b'\n' in buffer:
                        line,buffer=buffer.split(b'\n',1);command=line.decode().strip();requests.append(command)
                        if command.startswith('get '):peer.sendall(b'FlightController AircraftAttitude {yaw=12.5}\r\n')
                        elif command.startswith('unlisten '):
                            subscribed=False;peer.sendall(b'FlightController AircraftAttitude success\r\n')
                        elif command.startswith('listen '):
                            if reject_listen:peer.sendall(b"FlightController AircraftAttitude Cannot command 'LISTEN' on key.\r\n")
                            else:subscribed=True
                    if subscribed and push:peer.sendall(b'FlightController AircraftAttitude {yaw=13.5}\r\n')
            except OSError:pass
            finally:
                if peer:peer.close()
        worker=threading.Thread(target=serve,daemon=True);worker.start()
        def connect(address,timeout):
            self.assertEqual(address[1],9997)
            client=socket.create_connection(listener.getsockname(),timeout);clients.append(client);return client
        receiver=HeadingReceiver('local-test-only',connector=connect,on_event=lambda e,d:events.append((e,d)))
        receiver.start()
        def clean():
            receiver.stop();done.set();listener.close();worker.join(1.)
        self.addCleanup(clean)
        return receiver,requests,events,clients

    def wait_for(self,predicate):
        end=time.monotonic()+2.
        while not predicate() and time.monotonic()<end:time.sleep(.005)
        self.assertTrue(predicate())

    def test_listen_receives_push_and_enables_nodelay_using_only_read_commands(self):
        receiver,requests,events,clients=self.run_peer()
        self.wait_for(lambda:receiver.get() is not None and receiver.get()['yaw_deg']==13.5)
        self.assertEqual(clients[0].getsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY),1)
        self.assertEqual(sum(c.startswith('listen ') for c in requests),1)
        self.assertTrue(all(c.split()[0] in ('get','listen','unlisten') for c in requests))
        self.assertLess(receiver.get()['time'],receiver.get()['received_at'])
        receiver.stop()
        self.wait_for(lambda:sum(c.startswith('unlisten ') for c in requests)>=2)

    def test_rejected_listen_falls_back_without_reconnecting(self):
        receiver,requests,events,clients=self.run_peer(reject_listen=True)
        self.wait_for(lambda:sum(c.startswith('get ') for c in requests)>=3)
        self.assertEqual(len(clients),1);self.assertEqual(receiver.get()['yaw_deg'],12.5)
        self.assertTrue(any(e=='heading_listen_fallback' for e,d in events))

    def test_get_fallback_keeps_bounded_request_cadence(self):
        receiver,requests,events,clients=self.run_peer(reject_listen=True)
        self.wait_for(lambda:any(e=='heading_listen_fallback' for e,d in events))
        before=sum(c.startswith('get ') for c in requests)
        time.sleep(.25)
        count=sum(c.startswith('get ') for c in requests)-before
        self.assertGreater(count,0);self.assertLessEqual(count,7)

    def test_stationary_target_without_push_keeps_get_heartbeat(self):
        receiver,requests,events,clients=self.run_peer(push=False)
        self.wait_for(lambda:sum(c.startswith('get ') for c in requests)>=3)
        # Request receipt is not response publication on the receiver thread.
        # Observe freshness atomically with the sample instead of inspecting
        # the previous response immediately after the peer sees GET number 3.
        observed=[]
        def fresh_response():
            sample=receiver.get()
            if sample is None:return False
            age=time.monotonic()-sample['time']
            if 0<=age<.25:
                observed.append((sample,age));return True
            return False
        self.wait_for(fresh_response)
        self.assertEqual(len(clients),1)
        self.assertLess(observed[-1][1],.25)
        self.assertEqual(observed[-1][0]['yaw_deg'],12.5)

    def test_control_channel_uses_nodelay_without_changing_ack_contract(self):
        listener=socket.socket();listener.bind(('127.0.0.1',0));listener.listen(1)
        self.addCleanup(listener.close)
        client=socket.create_connection(listener.getsockname());peer,_=listener.accept()
        self.addCleanup(client.close);self.addCleanup(peer.close)
        channel=ControlChannel(client,lambda keys:None,lambda *args:None)
        self.assertTrue(channel.diagnostics()['tcp_nodelay'])
        self.assertEqual(channel.diagnostics()['acknowledgement_scope'],'android_handler_not_aircraft_motion')
        self.assertFalse(channel.neutral_acknowledged)


if __name__=='__main__':unittest.main()
