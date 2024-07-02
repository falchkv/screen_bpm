# This does not register lmscreen_save

import time

import numpy
from p06io import ClientReq, ClientSub





class MetadataTriggerer():
    def __init__(self, host, req_port, pub_port):
        self.host = host
        self.pub_port = pub_port
        self.req_port = req_port
        self.poll_rate = 0.2  # Hz
        self.t_triggered = previous_t_triggered = time.time() - 999
        self.set_trigger_info(self.t_triggered)

    def set_trigger_info(self, metadata):
        self.trigger_info = metadata

    def get_trigger_info(self):
        """

        Returns
        -------
        dict
            The trigger info
        """
        return self.trigger_info

    def start_monitoring(self):
        """


        :return:
        """
        poll_delay = 1.0 / self.poll_rate
        t0 = time.time()
        t_prev = t0
        sig_interval = 1.0  # time between fake triggers
        t_next = t_prev + sig_interval
        detector_name = 'xrayeye_eh1'
        print(self.host)
        with ClientSub(self.host, self.pub_port) as client_sub:
            topic = 'scan_finished'
            client_sub.add_subscription_topic(topic)

            while True:
                print('listening')
                message = client_sub.receive_message(timeout=poll_delay)
                print(message)
                #metadata = client_sub.receive(timeout=poll_delay)
                #print(metadata)
                #self.t_triggered = time.time()
                #self.set_trigger_info(metadata)




if __name__ == '__main__':
    topology_manager_host = 'haspp06.desy.de'
    topology_manager_pub_port = 13345
    with ClientReq(topology_manager_host, topology_manager_pub_port) as tm_req:
        print('about to request')
        info = tm_req.send_receive_message(['info', ['metadata_server']], timeout=1)
        if len(info) > 1:
            raise ValueError(
                'More than one metadataserver connection available:' +
                ' '.join([key for key in info.keys()]))
        from pprint import pprint
        pprint(info)

        host = info[list(info.keys())[0]]['connections'][ 'data_publisher']['host']
        req_port = info[list(info.keys())[0]]['connections']['data_requester']['port']
        pub_port = info[list(info.keys())[0]]['connections']['data_publisher']['port']
    print(host, req_port, pub_port)
    triggerer = MetadataTriggerer(host, req_port, pub_port)
    triggerer.start_monitoring()
