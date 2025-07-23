#!/usr/bin/env python3
"""
Quick script to list hydrophone names and their deployment locations to a txt file.
Uses the same logic as the existing data downloader functionality.

Usage:
    python list_hydrophone_locations.py [output_file.txt]
    
    Default output file: hydrophone_locations.txt
"""

import os
import sys
from collections import defaultdict
from datetime import datetime

# add the utils directory to the path so we can import the deployment checker
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from utils.data.deployment_checker import HydrophoneDeploymentChecker


def list_hydrophone_locations(onc_token, output_file="hydrophone_locations.txt"):
    """
    List all hydrophones and their deployment locations.
    
    Args:
        onc_token: ONC API token
        output_file: Path to output txt file
    """
    print("🌊 Fetching hydrophone deployment locations from ONC...")
    
    # init the deployment checker
    checker = HydrophoneDeploymentChecker(onc_token, debug=False)
    
    # get all deployments
    deployments = checker.get_all_hydrophone_deployments()
    
    if not deployments:
        print("❌ No deployments found!")
        return
    
    # group by device code
    device_deployments = defaultdict(list)
    for dep in deployments:
        device_deployments[dep.device_code].append(dep)
    
    # write to file
    with open(output_file, 'w') as f:
        f.write(f"ONC Hydrophone Deployment Locations\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Found {len(device_deployments)} hydrophones with {len(deployments)} total deployments\n")
        f.write("=" * 80 + "\n")
        
        # sort devices alphabetically
        for device_code in sorted(device_deployments.keys()):
            deps = device_deployments[device_code]
            f.write(f"\n{device_code}\n")
            
            # sort deployments by start date
            deps.sort(key=lambda x: x.begin_date)
            
            for i, dep in enumerate(deps, 1):
                end_str = dep.end_date.strftime('%Y-%m-%d') if dep.end_date else "ongoing"
                
                f.write(f"   Deployment {i}:\n")
                f.write(f"     Location: {dep.location_name or dep.location_code}\n")
                f.write(f"     Period: {dep.begin_date.strftime('%Y-%m-%d')} to {end_str}\n")
                
                if dep.latitude and dep.longitude:
                    f.write(f"     Coords: {dep.latitude:.4f}°N, {abs(dep.longitude):.4f}°W\n")
                
                if dep.depth:
                    f.write(f"     Depth: {dep.depth}m\n")
                
                f.write("\n")
    
    print(f"✅ Results saved to {output_file}")


def main():
    """Main function to run the script."""
    # check for ONC token
    onc_token = os.getenv('ONC_TOKEN')
    
    if not onc_token:
        print("❌ ONC_TOKEN environment variable not set!")
        print("   Please set it with: export ONC_TOKEN='your_token_here'")
        print("   Or get one from: https://data.oceannetworks.ca/Profile")
        return 1
    
    # check for output file argument
    output_file = sys.argv[1] if len(sys.argv) > 1 else "hydrophone_locations.txt"
    
    try:
        list_hydrophone_locations(onc_token, output_file)
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 