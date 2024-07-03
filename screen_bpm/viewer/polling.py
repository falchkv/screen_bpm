from tango import DeviceProxy


class TangoPoller:

    def __init__(self, poll_targets=None):
        """
        Parameters
        ----------
        poll_targets : dict
            Contains tuples of tango address and attribute to be polled.
            E.g. ('hasp029rack:10000/p06/tangovimba/test.02', 'Image8')
        """
        self.poll_targets = {}
        self.device_proxies = {}

        if poll_targets is not None:
            self.set_poll_targets(poll_targets)

    def set_poll_targets(self, poll_targets):
        """
        Sets which tango attributes to poll.

        poll_targets : iterable
            Contains tuples of tango address and attribute to be polled.
            E.g. ('hasp029rack:10000/p06/tangovimba/test.02', 'Image8')
        """
        # Check validity and create device proxies.
        self.poll_targets = poll_targets
        for key, poll_target in self.poll_targets.items():
            if len(poll_target) != 2:
                raise ValueError('Invalid poll_target: {}'.format(poll_target))
            else:
                host_port_device, attribute_name = poll_target
                device_proxy = DeviceProxy(host_port_device)
                print(device_proxy)
                attribute_list = device_proxy.get_attribute_list()
                if attribute_name not in attribute_list:
                    raise ValueError('Attribute {} is not in attribute list'
                                     ' for {}.\n Availabe attributes are: {}'
                                     .format(attribute_name,
                                             host_port_device, attribute_list)
                                     )
                else:
                    self.device_proxies[key] = device_proxy

    def poll_by_key(self, key):
        """
        polls a poll_target at given index.

        Parameters
        ----------
        key : int
            key of poll_target

        Returns
        -------
        any
            Results of polling
        """
        attribute_name = self.poll_targets[key][1]
        device_proxy = self.device_proxies[key]
        return device_proxy.read_attribute(attribute_name).value

    def poll_all(self):
        """
        polls all the poll_target by successively calling self.poll_by_key()

        Returns
        -------
        any
            Results of polling
        """
        result = {}
        for key in self.poll_targets:
            result[key] = self.poll_by_key(key)

        return result

    def is_moving(self):
        """
        Checks state of all poll targets, and returns True if any of them are
        in 'MOVING' state.

        Returns
        -------
        bool
            True if any of self.poll_targets are moving
        """
        is_moving = False
        for key, device_proxy in self.device_proxies.items():
            if str(device_proxy.State()) == 'MOVING':
                is_moving = True
                break

        return is_moving


if __name__ == '__main__':
    to_poll = {
        'mscope': ('haspp06mc01:10000/p06/mscope/mi.01', 'Zoom'),
    }
    poller = TangoPoller(poll_targets=to_poll)
    res = poller.poll_all()
    print(res)
