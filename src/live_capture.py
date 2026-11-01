"""
NetSentinels: Live Packet Capture

Captures a maximum of 20 packets or runs for 30 seconds.
Counts packets without storing the captured packet list.
"""

from datetime import datetime

from scapy.all import sniff, IP, IPv6, TCP, UDP

packet_count = 0


def process_packet(packet):
    """Print a summary of each packet received."""

    global packet_count
    packet_count += 1

    timestamp = datetime.now().strftime("%H:%M:%S")

    source = "Unknown"
    destination = "Unknown"
    protocol = "Other"
    source_port = "-"
    destination_port = "-"

    if IP in packet:
        source = packet[IP].src
        destination = packet[IP].dst
    elif IPv6 in packet:
        source = packet[IPv6].src
        destination = packet[IPv6].dst

    if TCP in packet:
        protocol = "TCP"
        source_port = packet[TCP].sport
        destination_port = packet[TCP].dport
    elif UDP in packet:
        protocol = "UDP"
        source_port = packet[UDP].sport
        destination_port = packet[UDP].dport

    print(
        f"[{timestamp}] "
        f"{protocol:5} | "
        f"{source}:{source_port} -> "
        f"{destination}:{destination_port} | "
        f"Size: {len(packet)} bytes"
    )


def main():
    global packet_count
    packet_count = 0

    print("=" * 65)
    print("NETSENTINELS - LIVE PACKET CAPTURE TEST")
    print("=" * 65)
    print("Capturing up to 20 packets, for at most 30 seconds.")
    print("Open a website to generate network traffic.")
    print("Press Ctrl+C to stop early.")
    print("-" * 65)

    try:
        sniff(
            prn=process_packet,
            count=20,
            timeout=30,
            store=False,
        )

    except KeyboardInterrupt:
        print("\nCapture stopped by user.")

    except (PermissionError, OSError) as exc:
        print(f"\nPacket capture failed: {exc}")
        return

    finally:
        print("-" * 65)
        print(f"Packets processed: {packet_count}")

    if packet_count:
        print("Live packet capture is working.")
    else:
        print(
            "No packets were processed. Check Npcap, "
            "permissions, and the network interface."
        )


if __name__ == "__main__":
    main()
