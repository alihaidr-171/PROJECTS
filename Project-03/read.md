# Secure Enterprise Network Infrastructure

## Project Overview

This project focuses on the design, implementation, security, and testing of a simulated enterprise network infrastructure using GNS3, Cisco routers, pfSense, Ubuntu Server, Kali Linux, and OpenVPN.

The network was designed to simulate an enterprise/ISP environment containing multiple autonomous systems, redundant company networks, a security firewall, a DMZ, internal services, remote-access VPN, and security monitoring.

The project combines enterprise networking and cybersecurity concepts into a single practical environment.

---

## Objectives

- Design a realistic enterprise network topology using GNS3.
- Implement dynamic routing using OSPF.
- Implement inter-AS routing using eBGP.
- Configure HSRP for gateway redundancy and failover.
- Configure BGP routing policies using prefix lists, route maps, and local preference.
- Deploy pfSense as the main firewall and security gateway.
- Implement NAT/PAT for outbound connectivity.
- Create a DMZ for public-facing services.
- Deploy and configure Ubuntu Server services.
- Implement OpenVPN for secure remote access.
- Implement centralized logging using rsyslog.
- Perform security testing using Kali Linux.
- Verify firewall rules, service exposure, NAT, routing, and DMZ isolation.

---

## Network Architecture

```text
                                INTERNET
                                   |
                                   |
                              +---------+
                              | pfSense |
                              | Firewall|
                              +----+----+
                                   |
                              WAN Network
                            192.168.10.0/24
                                   |
                                  R1
                                   |
                                  R2
                              /          \
                            R3            R4
                             \            /
                              \  OSPF    /
                               \        /
                                -------
                                  |
                    +-------------+-------------+
                    |                           |
                Company A                   Company B
                AS 65010                    AS 65020
                 /    \                      /    \
                R5    R7                    R6    R8
              Active Standby             Active Standby
                HSRP                       HSRP
