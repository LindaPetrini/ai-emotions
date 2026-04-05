#!/usr/bin/env python3
"""Hyperstack GPU VM management for ai-emotions-v2.

Usage:
    python scripts/hyperstack.py create    # Create GPU VM
    python scripts/hyperstack.py status    # Check VM status
    python scripts/hyperstack.py delete    # Delete VM
    python scripts/hyperstack.py ssh       # Print SSH command
"""

import argparse
import json
import sys
import urllib.request
import urllib.error

API_KEY = "9bb0b166-37f6-424e-9c1c-fa4ac717b801"
API_BASE = "https://infrahub-api.nexgencloud.com/v1/core/virtual-machines"
SSH_KEY = "~/.ssh/id_rsa_hyperstack"
VM_NAME = "ai-emotions-v2"

# Default: cheapest GPU with enough VRAM for 8B models (~16GB)
DEFAULT_FLAVOR = "n1-RTX-A5000x1"
DEFAULT_IMAGE = "Ubuntu Server 22.04 LTS R535 CUDA 12.2"
DEFAULT_KEY_NAME = "linda-key"


def _request(method, url, data=None):
    """Make API request."""
    headers = {
        "api_key": API_KEY,
        "Content-Type": "application/json",
    }

    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode()
        print(f"HTTP {e.code}: {error_body}")
        sys.exit(1)


def create_vm():
    """Create a new GPU VM."""
    data = {
        "name": VM_NAME,
        "environment_name": "CANADA-1",
        "image_name": DEFAULT_IMAGE,
        "flavor_name": DEFAULT_FLAVOR,
        "key_name": DEFAULT_KEY_NAME,
        "assign_floating_ip": True,
    }

    print(f"Creating VM: {VM_NAME}")
    print(f"  Flavor: {DEFAULT_FLAVOR}")
    print(f"  Image: {DEFAULT_IMAGE}")

    result = _request("POST", API_BASE, data)
    print(json.dumps(result, indent=2))

    if "instance" in result:
        vm = result["instance"]
        print(f"\nVM created: {vm.get('name', VM_NAME)}")
        print(f"  ID: {vm.get('id')}")
        print(f"  Status: {vm.get('status')}")
        print("  Wait a few minutes for it to boot, then check with: python scripts/hyperstack.py status")


def get_vms():
    """Get all VMs."""
    result = _request("GET", API_BASE)
    return result.get("instances", result.get("virtual_machines", []))


def status():
    """Show status of all VMs."""
    vms = get_vms()
    if not vms:
        print("No VMs found.")
        return

    for vm in vms:
        name = vm.get("name", "unknown")
        status = vm.get("status", "unknown")
        ip = None

        # Extract floating IP
        floating_ip = vm.get("floating_ip")
        if floating_ip:
            ip = floating_ip if isinstance(floating_ip, str) else floating_ip.get("ip")

        if not ip:
            # Try fixed IP
            fixed_ip = vm.get("fixed_ip")
            if fixed_ip:
                ip = fixed_ip if isinstance(fixed_ip, str) else fixed_ip.get("ip")

        print(f"  {name}: {status} | IP: {ip or 'pending'}")

        if ip and name == VM_NAME:
            print(f"\n  SSH: ssh -i {SSH_KEY} ubuntu@{ip}")


def delete_vm():
    """Delete the VM."""
    vms = get_vms()
    target = None
    for vm in vms:
        if vm.get("name") == VM_NAME:
            target = vm
            break

    if not target:
        print(f"VM '{VM_NAME}' not found.")
        return

    vm_id = target.get("id")
    print(f"Deleting VM: {VM_NAME} (ID: {vm_id})")
    result = _request("DELETE", f"{API_BASE}/{vm_id}")
    print("VM deleted.")


def ssh_cmd():
    """Print SSH command."""
    vms = get_vms()
    for vm in vms:
        if vm.get("name") == VM_NAME:
            floating_ip = vm.get("floating_ip")
            ip = None
            if floating_ip:
                ip = floating_ip if isinstance(floating_ip, str) else floating_ip.get("ip")
            if ip:
                print(f"ssh -i {SSH_KEY} ubuntu@{ip}")
                return
    print("VM not found or no IP assigned.")


def main():
    parser = argparse.ArgumentParser(description="Hyperstack GPU VM management")
    parser.add_argument("action", choices=["create", "status", "delete", "ssh"])
    args = parser.parse_args()

    actions = {
        "create": create_vm,
        "status": status,
        "delete": delete_vm,
        "ssh": ssh_cmd,
    }
    actions[args.action]()


if __name__ == "__main__":
    main()
