#!/bin/bash
while true; do
	sudo hping3 --count 100 --udp 192.168.113.62 -p 80 -i u1000
	sleep 10
done
