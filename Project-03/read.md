Secure Enterprise Network Infrastructure

A hands-on enterprise networking and cybersecurity project designed and implemented using GNS3, Cisco routers, pfSense, Ubuntu Server, Kali Linux, and OpenVPN.

The project simulates a multi-company enterprise/ISP environment with dynamic routing, high availability, firewall security, DMZ services, VPN access, NAT, DNS, centralized logging, and security testing.

Project Architecture
                              Internet
                                  |
                              pfSense
                             Firewall
                                  |
                         +--------+--------+
                         |                 |
                      Internal             DMZ
                         |            192.168.60.0/24
                    Enterprise              |
                         |             Ubuntu Server
                    ISP / Core             |
                   OSPF + BGP        HTTP / HTTPS
                         |            DNS / FTP
                  +------+------+\     SSH / Syslog
                  |             |
              Company A      Company B
              AS 65010       AS 65020
                  |             |
             R5 / R7        R6 / R8
                HSRP           HSRP
Main Technologies
Technology	Purpose
GNS3	Network simulation
Cisco IOS	Enterprise/core routing
OSPF	Internal dynamic routing
eBGP	Inter-AS routing
HSRP	Gateway redundancy
pfSense	Firewall, NAT, VPN, DNS integration
NAT/PAT	Internet address translation
DMZ	Segmentation of public-facing services
OpenVPN	Secure remote access
Ubuntu Server	Network and application services
BIND9	DNS
Apache	HTTP/HTTPS
vsftpd	FTP
OpenSSH	Secure administration
rsyslog	Centralized logging
Kali Linux	Security testing
Routing and High Availability
OSPF

OSPF is used for dynamic routing within the core network and for distributing internal routes.

Router loopbacks are used as stable router IDs.

R1 → 1.1.1.1
R2 → 2.2.2.2
R3 → 3.3.3.3
R4 → 4.4.4.4
R5 → 5.5.5.5
R6 → 6.6.6.6
R7 → 7.7.7.7
R8 → 8.8.8.8
BGP

Three Autonomous Systems are used:

AS 65001 → ISP/Core
AS 65010 → Company A
AS 65020 → Company B

BGP policies use:

Prefix Lists
Route Maps
Local Preference
AS-PATH

Primary and backup BGP paths were tested by shutting down the primary routers.

HSRP

HSRP provides first-hop gateway redundancy.

Company A
R5 → Active
R7 → Standby


Company B
R6 → Active
R8 → Standby

Failover was tested by shutting down the active routers and verifying that the standby routers became active.

pfSense Firewall

pfSense acts as the main security gateway.

Interfaces
WAN  → 192.168.10.10
LAN  → 192.168.75.2
OPT1 → 192.168.50.1
OPT2 → 192.168.60.1

Key pfSense functions:

Firewall filtering
Outbound NAT
Port forwarding
DNS Resolver
DNS Domain Overrides
OpenVPN
Firewall logging
DMZ segmentation
DMZ

The DMZ uses:

Network:       192.168.60.0/24
pfSense OPT2:  192.168.60.1
Ubuntu DMZ:    192.168.60.10

Public-facing services are placed in the DMZ and exposed only through specific firewall/NAT rules.

Public Web Services
WAN TCP/80  → 192.168.60.10:80
WAN TCP/443 → 192.168.60.10:443

HTTP and HTTPS were successfully tested from Kali Linux.

Ubuntu Server

The Ubuntu server provides multiple services.

Network Interfaces
ens33 → 192.168.50.21
ens37 → 192.168.60.10
Services

Apache

HTTP: TCP/80
HTTPS: TCP/443
Self-signed SSL/TLS certificate for the lab

BIND9

Internal DNS zone:

isp.lab

Example record:

web.isp.lab → 192.168.60.10

FTP

vsftpd
TCP/21
Local authentication
Anonymous access disabled

SSH

TCP/22
Restricted through pfSense firewall policy

Syslog

rsyslog
UDP/514
Remote logs tested from Kali Linux
DNS Architecture

DNS forwarding was configured through pfSense.

Kali
192.168.10.13
      |
      | DNS
      v
pfSense
192.168.10.10
      |
      | Domain Override
      v
Ubuntu BIND9
192.168.50.21
      |
      v
web.isp.lab
192.168.60.10

The pfSense DNS Resolver also uses an access-control rule to restrict DNS access to the authorized Kali client.

OpenVPN

OpenVPN was configured on pfSense for remote access.

Protocol: UDP
Port: 1194
Tunnel Network: 10.10.10.0/24
Authentication: SSL/TLS + User Authentication

A VPN client successfully established a connection and received a tunnel address.

NAT/PAT

Cisco PAT was configured on R1 for outbound enterprise traffic.

Example:

Private:
192.168.20.100


Translated:
192.168.50.10

The translation was verified using:

show ip nat translations
show ip nat statistics

pfSense also provides outbound NAT toward the external network.

Security Testing

Kali Linux was used to verify the implementation.

Tools Used
Nmap
Netcat
curl
dig
ping
logger
Testing Performed
OSPF neighbor verification
BGP path verification
HSRP failover testing
NAT/PAT verification
DMZ service scanning
WAN port exposure testing
DNS resolution testing
HTTP/HTTPS testing
FTP testing
SSH testing
Remote Syslog testing
OpenVPN testing
DMZ-to-internal isolation testing
Firewall log verification

Example DMZ scan:

nmap -Pn -sT -p 21,22,80,443 192.168.60.10

Expected intentionally exposed services:

21/tcp   FTP
22/tcp   SSH
80/tcp   HTTP
443/tcp  HTTPS

Unnecessary ports were also scanned and verified as closed or filtered.

Security Design

The project follows several security principles:

Least privilege
Network segmentation
Controlled service exposure
Firewall-based access control
Secure remote administration
Encrypted remote access
Centralized logging
Security verification

The DMZ was tested to confirm that traffic routed through the DMZ interface toward the protected internal network was blocked.

Project Results

The following components were successfully implemented and tested:

OSPF
eBGP
HSRP
BGP routing policies
pfSense firewall
NAT/PAT
DMZ
HTTP/HTTPS
BIND9 DNS
FTP
SSH
Syslog
OpenVPN
Firewall logging
Security testing
Limitations and Future Improvements

The current Ubuntu server is dual-homed:

ens33 → 192.168.50.21
ens37 → 192.168.60.10

A future version should use a dedicated single-homed DMZ server to provide stronger isolation.

Other planned improvements include:

VLAN segmentation with an IOS Layer-2 switch
Dedicated DMZ server
SIEM integration
Centralized security monitoring
Trusted SSL/TLS certificates
More granular firewall policies
Additional redundancy
Repository Contents

A recommended repository structure:

secure-enterprise-network/
│
├── README.md
├── topology/
│   └── enterprise-network.gns3
│
├── documentation/
│   └── project-report.pdf
│
├── screenshots/
│   ├── topology/
│   ├── routing/
│   ├── pfsense/
│   ├── dmz/
│   ├── vpn/
│   └── security-testing/
│
└── configs/
    ├── router-configs/
    ├── pfsense/
    └── ubuntu/

Note: Do not upload sensitive VPN private keys, certificates containing private material, passwords, or other secrets to a public repository.

Author

Ali Haider
Cyber Security Student
AIR University

Disclaimer

This project was developed as an educational cybersecurity and networking laboratory. All security testing was performed against the author's own simulated infrastructure and authorized lab environment.
