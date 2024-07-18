package main

import (
	"bufio"
	"net"
	"os/exec"
	"strings"
)

func getDefaultIPv6Address() (string, error) {
	ifaceName, err := getDefaultInterface()
	if err != nil {
		return "", err
	}

	iface, err := net.InterfaceByName(ifaceName)
	if err != nil {
		return "", err
	}

	addrs, err := iface.Addrs()
	if err != nil {
		return "", err
	}

	for _, addr := range addrs {
		if ipNet, ok := addr.(*net.IPNet); ok && !ipNet.IP.IsLoopback() {
			if ipNet.IP.To4() == nil {
				return ipNet.IP.String(), nil
			}
		}
	}

	return "", nil // Return empty string if no IPv6 address found
}

func getDefaultInterface() (string, error) {
	cmd := exec.Command("ip", "route", "show", "default")
	output, err := cmd.Output()
	if err != nil {
		return "", err
	}

	scanner := bufio.NewScanner(strings.NewReader(string(output)))
	for scanner.Scan() {
		line := scanner.Text()
		fields := strings.Fields(line)
		for i, field := range fields {
			if field == "dev" && i+1 < len(fields) {
				return fields[i+1], nil
			}
		}
	}

	return "", scanner.Err()
}
