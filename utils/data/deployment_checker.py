"""
ONC Hydrophone Deployment Date Checker

This module provides functionality to check deployment dates for hydrophones
from Ocean Networks Canada (ONC) and helps users choose appropriate dates
for downloading hydrophone data.

Based on functionality from: https://github.com/Spiffical/hydrophonedatarequests
"""
import logging
import time
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional, Union
import concurrent.futures
from dataclasses import dataclass
import contextlib
from io import StringIO

try:
    from dateutil import parser as dtparse
    from dateutil.tz import gettz, UTC
except ImportError:
    raise ImportError("ERROR: 'python-dateutil' library not found. Please install it: pip install python-dateutil")

try:
    from onc import ONC
    from requests.exceptions import HTTPError
except ImportError:
    raise ImportError("ERROR: 'onc-python' library not found. Please install it: pip install onc-python")


@dataclass
class DeploymentInfo:
    """Information about a hydrophone deployment."""
    device_code: str
    location_code: str
    location_name: str
    begin_date: datetime
    end_date: Optional[datetime]
    latitude: float
    longitude: float
    depth: Optional[float]
    citation: Optional[str]
    has_data: bool = False


class HydrophoneDeploymentChecker:
    """Check deployment dates and data availability for ONC hydrophones."""
    
    def __init__(self, onc_token: str, debug: bool = False):
        """
        Initialize the deployment checker.
        
        Args:
            onc_token: ONC API token
            debug: Enable debug logging
        """
        self.onc = ONC(onc_token, showInfo=debug)
        # Best-effort: quiet the ONC client if it supports these toggles
        for attr, value in (('showInfo', False), ('showWarnings', False), ('showErrors', False)):
            try:
                setattr(self.onc, attr, value)
            except Exception:
                pass
        self.debug = debug
        self._location_cache = {}
        
        # Setup logging
        log_level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(level=log_level, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
        
    def get_all_hydrophone_deployments(self) -> List[DeploymentInfo]:
        """
        Get all hydrophone deployments from ONC.
        
        Returns:
            List of DeploymentInfo objects for all deployments
        """
        logging.info("Fetching hydrophone devices from ONC...")
        
        # Get all hydrophone devices
        hydrophones = self.onc.getDevices({"deviceCategoryCode": "HYDROPHONE"})
        if not isinstance(hydrophones, list):
            raise ValueError("Unexpected response format from getDevices")
        
        if not hydrophones:
            raise ValueError("No hydrophone devices found")
        
        logging.info(f"Found {len(hydrophones)} hydrophone device(s)")
        
        # Get location information for mapping
        self._build_location_cache()
        
        # Get deployments for all hydrophones in parallel
        all_deployments = self._get_deployments_parallel(hydrophones)
        
        # Convert to DeploymentInfo objects
        deployment_infos = []
        for dep in all_deployments:
            try:
                deployment_info = self._parse_deployment(dep)
                if deployment_info:
                    deployment_infos.append(deployment_info)
            except Exception as e:
                if self.debug:
                    logging.warning(f"Error parsing deployment: {e}")
                continue
        
        logging.info(f"Found {len(deployment_infos)} valid deployment(s)")
        return deployment_infos
    
    def find_deployments_by_time_range(self, 
                                     start_date: Union[str, datetime], 
                                     end_date: Union[str, datetime],
                                     timezone_str: str = 'UTC') -> List[DeploymentInfo]:
        """
        Find deployments that overlap with a specific time range.
        
        Args:
            start_date: Start date (string or datetime)
            end_date: End date (string or datetime)
            timezone_str: Timezone for date interpretation
            
        Returns:
            List of deployments that overlap with the time range
        """
        # Parse dates
        if isinstance(start_date, str):
            start_dt = dtparse.parse(start_date)
        else:
            start_dt = start_date
            
        if isinstance(end_date, str):
            end_dt = dtparse.parse(end_date)
        else:
            end_dt = end_date
        
        # Handle timezone
        if timezone_str != 'UTC':
            tz = gettz(timezone_str)
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=tz)
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=tz)
        
        # Convert to UTC
        start_utc = start_dt.astimezone(UTC) if start_dt.tzinfo else start_dt.replace(tzinfo=UTC)
        end_utc = end_dt.astimezone(UTC) if end_dt.tzinfo else end_dt.replace(tzinfo=UTC)
        
        logging.info(f"Searching for deployments overlapping: {start_utc} to {end_utc}")
        
        # Get all deployments
        all_deployments = self.get_all_hydrophone_deployments()
        
        # Filter by time overlap
        overlapping = []
        for dep in all_deployments:
            # Check if deployment overlaps with requested time range
            if (dep.begin_date <= end_utc) and (not dep.end_date or dep.end_date >= start_utc):
                overlapping.append(dep)
        
        logging.info(f"Found {len(overlapping)} overlapping deployment(s)")
        return overlapping
    
    def check_data_availability(self, 
                              deployments: List[DeploymentInfo], 
                              start_date: datetime, 
                              end_date: datetime,
                              check_archive: bool = False) -> List[DeploymentInfo]:
        """
        Check data availability for deployments in the specified time range.
        
        Args:
            deployments: List of deployments to check
            start_date: Start date for data availability check
            end_date: End date for data availability check
            check_archive: If True, check for archive files, otherwise check data products
            
        Returns:
            List of deployments with data availability marked
        """
        logging.info(f"Checking data availability for {len(deployments)} deployment(s)...")
        
        # Group by device code for efficiency
        device_to_deployments = defaultdict(list)
        for dep in deployments:
            device_to_deployments[dep.device_code].append(dep)
        
        # Check availability for each device
        if check_archive:
            device_availability = self._check_archive_availability_parallel(
                list(device_to_deployments.keys()), start_date, end_date)
        else:
            device_availability = self._check_product_availability_parallel(
                list(device_to_deployments.keys()))
        
        # Update deployment info with availability
        available_deployments = []
        for device_code, has_data in device_availability.items():
            for dep in device_to_deployments[device_code]:
                dep.has_data = has_data
                if has_data:
                    available_deployments.append(dep)
        
        logging.info(f"Found {len(available_deployments)} deployment(s) with available data")
        return available_deployments
    
    def get_deployment_date_ranges(self, device_codes: Optional[List[str]] = None) -> Dict[str, List[Tuple[datetime, Optional[datetime]]]]:
        """
        Get deployment date ranges for specific device codes or all hydrophones.
        
        Args:
            device_codes: Optional list of device codes to check. If None, checks all.
            
        Returns:
            Dictionary mapping device codes to list of (start_date, end_date) tuples
        """
        all_deployments = self.get_all_hydrophone_deployments()
        
        # Filter by device codes if specified
        if device_codes:
            all_deployments = [dep for dep in all_deployments if dep.device_code in device_codes]
        
        # Group by device code
        date_ranges = defaultdict(list)
        for dep in all_deployments:
            date_ranges[dep.device_code].append((dep.begin_date, dep.end_date))
        
        # Sort date ranges
        for device_code in date_ranges:
            date_ranges[device_code].sort(key=lambda x: x[0])
        
        return dict(date_ranges)
    
    def print_deployment_summary(self, deployments: List[DeploymentInfo], show_data_availability: bool = True):
        """
        Print a formatted summary of deployments.
        
        Args:
            deployments: List of deployments to summarize
            show_data_availability: Whether to show data availability status
        """
        if not deployments:
            print("No deployments found.")
            return
        
        print(f"\n{'='*80}")
        print(f"HYDROPHONE DEPLOYMENT SUMMARY ({len(deployments)} deployments)")
        print(f"{'='*80}")
        
        # Group by location for better organization
        by_location = defaultdict(list)
        for dep in deployments:
            by_location[dep.location_name or dep.location_code].append(dep)
        
        for location, deps in sorted(by_location.items()):
            print(f"\n📍 Location: {location}")
            print("-" * 60)
            
            for dep in sorted(deps, key=lambda x: x.begin_date):
                end_str = dep.end_date.strftime('%Y-%m-%d') if dep.end_date else "ongoing"
                data_status = " ✅ Has Data" if show_data_availability and dep.has_data else " ❌ No Data" if show_data_availability else ""
                
                print(f"  🔹 {dep.device_code}")
                print(f"     Period: {dep.begin_date.strftime('%Y-%m-%d')} to {end_str}{data_status}")
                if dep.depth:
                    print(f"     Depth: {dep.depth}m")
                if dep.latitude and dep.longitude:
                    print(f"     Location: {dep.latitude:.4f}°N, {dep.longitude:.4f}°W")
                print()
    
    def find_best_deployments_for_date_range(self, 
                                           start_date: Union[str, datetime], 
                                           end_date: Union[str, datetime],
                                           timezone_str: str = 'UTC',
                                           min_coverage_days: int = 1) -> List[DeploymentInfo]:
        """
        Find the best deployments that cover a specific date range.
        
        Args:
            start_date: Desired start date
            end_date: Desired end date
            timezone_str: Timezone for date interpretation
            min_coverage_days: Minimum days of coverage required
            
        Returns:
            List of best deployments sorted by coverage quality
        """
        # Parse and convert dates
        if isinstance(start_date, str):
            start_dt = dtparse.parse(start_date)
        else:
            start_dt = start_date
            
        if isinstance(end_date, str):
            end_dt = dtparse.parse(end_date)
        else:
            end_dt = end_date
        
        # Handle timezone
        if timezone_str != 'UTC':
            tz = gettz(timezone_str)
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=tz)
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=tz)
        
        # Convert to UTC
        start_utc = start_dt.astimezone(UTC) if start_dt.tzinfo else start_dt.replace(tzinfo=UTC)
        end_utc = end_dt.astimezone(UTC) if end_dt.tzinfo else end_dt.replace(tzinfo=UTC)
        
        # Find overlapping deployments
        overlapping = self.find_deployments_by_time_range(start_utc, end_utc, 'UTC')
        
        # Check data availability
        available = self.check_data_availability(overlapping, start_utc, end_utc)
        
        # Calculate coverage quality for each deployment
        scored_deployments = []
        for dep in available:
            # Calculate overlap period
            overlap_start = max(start_utc, dep.begin_date)
            overlap_end = min(end_utc, dep.end_date) if dep.end_date else end_utc
            
            if overlap_end > overlap_start:
                coverage_days = (overlap_end - overlap_start).days
                total_requested_days = (end_utc - start_utc).days
                
                if coverage_days >= min_coverage_days:
                    coverage_ratio = coverage_days / max(total_requested_days, 1)
                    scored_deployments.append((dep, coverage_ratio, coverage_days))
        
        # Sort by coverage ratio (best first)
        scored_deployments.sort(key=lambda x: x[1], reverse=True)
        
        return [dep for dep, _, _ in scored_deployments]
    
    def _build_location_cache(self):
        """Build cache of location information."""
        try:
            locations = self.onc.getLocations({})
            for loc in locations:
                if isinstance(loc, dict):
                    code = loc.get('locationCode')
                    if code:
                        self._location_cache[code] = loc
        except Exception as e:
            logging.warning(f"Failed to build location cache: {e}")
    
    def _get_deployments_parallel(self, hydrophones: List[Dict], max_workers: int = 10) -> List[Dict]:
        """Fetch deployments for multiple hydrophones in parallel."""
        def fetch_device_deployments(device: Dict) -> List[Dict]:
            device_code = device.get("deviceCode")
            if not device_code:
                return []
            
            try:
                device_deployments = self.onc.getDeployments({"deviceCode": device_code})
                if not isinstance(device_deployments, list):
                    logging.warning(f"Unexpected response type for {device_code} deployments")
                    return []
                
                # Add device info to each deployment
                for dep in device_deployments:
                    if isinstance(dep, dict):
                        dep.update(device)
                
                return device_deployments
            except HTTPError as http_err:
                if http_err.response is not None and http_err.response.status_code == 404:
                    if self.debug:
                        logging.debug(f"No deployments found (404) for device {device_code}")
                    return []
                raise
        
        all_deployments = []
        completed = 0
        total = len(hydrophones)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_device = {executor.submit(fetch_device_deployments, device): device 
                              for device in hydrophones}
            
            for future in concurrent.futures.as_completed(future_to_device):
                completed += 1
                print(f"\rFetching deployments: {completed}/{total}", end="", flush=True)
                
                try:
                    deployments = future.result()
                    if deployments:
                        all_deployments.extend(deployments)
                except Exception as e:
                    if self.debug:
                        logging.error(f"Error fetching deployments: {e}")
        
        print()  # New line after progress
        return all_deployments
    
    def _parse_deployment(self, deployment_dict: Dict) -> Optional[DeploymentInfo]:
        """Parse a deployment dictionary into a DeploymentInfo object."""
        try:
            device_code = deployment_dict.get('deviceCode')
            if not device_code:
                return None
            
            # Parse dates
            begin_str = deployment_dict.get('begin')
            if not begin_str:
                return None
            begin_date = dtparse.parse(begin_str)
            
            end_str = deployment_dict.get('end')
            end_date = dtparse.parse(end_str) if end_str else None
            
            # Get location info
            location_code = deployment_dict.get('locationCode', '')
            location_info = self._location_cache.get(location_code, {})
            location_name = location_info.get('locationName', '') or deployment_dict.get('locationName', '')
            
            # Get coordinates
            latitude = deployment_dict.get('lat') or location_info.get('lat', 0.0)
            longitude = deployment_dict.get('lon') or location_info.get('lon', 0.0)
            depth = deployment_dict.get('depth')
            
            citation = deployment_dict.get('citation')
            
            return DeploymentInfo(
                device_code=device_code,
                location_code=location_code,
                location_name=location_name,
                begin_date=begin_date,
                end_date=end_date,
                latitude=float(latitude) if latitude else 0.0,
                longitude=float(longitude) if longitude else 0.0,
                depth=float(depth) if depth else None,
                citation=citation
            )
        except Exception as e:
            if self.debug:
                logging.warning(f"Error parsing deployment: {e}")
            return None
    
    def _check_archive_availability_parallel(self, device_codes: List[str], 
                                           start_date: datetime, end_date: datetime,
                                           max_workers: int = 10) -> Dict[str, bool]:
        """Check archive file availability for multiple devices in parallel."""
        def check_device_files(device_code: str) -> Tuple[str, bool]:
            archive_filters = {
                'deviceCode': device_code,
                'dateFrom': start_date.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                'dateTo': end_date.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                'returnOptions': 'all'
            }
            try:
                # Suppress verbose ONC client stdout/stderr warnings during availability checks
                with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
                    list_result = self.onc.getArchivefile(filters=archive_filters, allPages=True)
                has_files = bool(list_result.get("files", []))
                return device_code, has_files
            except Exception as e:
                if self.debug:
                    logging.warning(f"Error checking files for device {device_code}: {e}")
                return device_code, False
        
        results = {}
        completed = 0
        total = len(device_codes)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_device = {executor.submit(check_device_files, device_code): device_code 
                              for device_code in device_codes}
            
            for future in concurrent.futures.as_completed(future_to_device):
                completed += 1
                print(f"\rChecking data availability: {completed}/{total}", end="", flush=True)
                
                try:
                    device_code, has_files = future.result()
                    results[device_code] = has_files
                except Exception as e:
                    if self.debug:
                        logging.error(f"Error checking device: {e}")
        
        print()  # New line after progress
        return results
    
    def _check_product_availability_parallel(self, device_codes: List[str], 
                                           max_workers: int = 10) -> Dict[str, bool]:
        """Check data product availability for multiple devices in parallel."""
        def check_device_products(device_code: str) -> Tuple[str, bool]:
            try:
                prod_opts = self.onc.getDataProducts({"deviceCode": device_code})
                has_products = bool(prod_opts and isinstance(prod_opts, list) and prod_opts)
                return device_code, has_products
            except Exception as e:
                if self.debug:
                    logging.warning(f"Error checking products for device {device_code}: {e}")
                return device_code, False
        
        results = {}
        completed = 0
        total = len(device_codes)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_device = {executor.submit(check_device_products, device_code): device_code 
                              for device_code in device_codes}
            
            for future in concurrent.futures.as_completed(future_to_device):
                completed += 1
                print(f"\rChecking data products: {completed}/{total}", end="", flush=True)
                
                try:
                    device_code, has_products = future.result()
                    results[device_code] = has_products
                except Exception as e:
                    if self.debug:
                        logging.error(f"Error checking device: {e}")
        
        print()  # New line after progress
        return results


def interactive_deployment_selector(checker: HydrophoneDeploymentChecker,
                                  start_date: Optional[Union[str, datetime]] = None,
                                  end_date: Optional[Union[str, datetime]] = None,
                                  timezone_str: str = 'UTC') -> List[DeploymentInfo]:
    """
    Interactive function to help users select deployments with data availability.
    
    Args:
        checker: HydrophoneDeploymentChecker instance
        start_date: Optional start date
        end_date: Optional end date
        timezone_str: Timezone for date interpretation
        
    Returns:
        List of selected deployments
    """
    print("🌊 ONC Hydrophone Deployment Selector")
    print("=" * 50)
    
    # Get date range if not provided
    if not start_date or not end_date:
        print("\nFirst, let's see what deployments are available...")
        all_deployments = checker.get_all_hydrophone_deployments()
        
        if not all_deployments:
            print("❌ No deployments found!")
            return []
        
        # Show summary of all deployments
        checker.print_deployment_summary(all_deployments, show_data_availability=False)
        
        # Get date range from user
        if not start_date:
            start_date = input("\n📅 Enter start date (YYYY-MM-DD or YYYY-MM-DD HH:MM): ").strip()
        if not end_date:
            end_date = input("📅 Enter end date (YYYY-MM-DD or YYYY-MM-DD HH:MM): ").strip()
    
    # Find best deployments for the date range
    print(f"\n🔍 Finding deployments for: {start_date} to {end_date} ({timezone_str})")
    best_deployments = checker.find_best_deployments_for_date_range(
        start_date, end_date, timezone_str)
    
    if not best_deployments:
        print("❌ No deployments with data found for the specified date range!")
        return []
    
    # Show available deployments with data
    print("\n✅ Found deployments with available data:")
    checker.print_deployment_summary(best_deployments, show_data_availability=True)
    
    # Let user select deployments
    print("\nSelect deployments to download (enter numbers separated by commas, or 'all'):")
    for i, dep in enumerate(best_deployments):
        end_str = dep.end_date.strftime('%Y-%m-%d') if dep.end_date else "ongoing"
        print(f"  [{i}] {dep.device_code} at {dep.location_name} ({dep.begin_date.strftime('%Y-%m-%d')} to {end_str})")
    
    selection = input("\nYour selection: ").strip().lower()
    
    if selection == 'all':
        return best_deployments
    
    try:
        indices = [int(x.strip()) for x in selection.split(',')]
        selected = [best_deployments[i] for i in indices if 0 <= i < len(best_deployments)]
        
        if selected:
            print(f"\n✅ Selected {len(selected)} deployment(s):")
            for dep in selected:
                print(f"  • {dep.device_code} at {dep.location_name}")
        else:
            print("❌ No valid selections made!")
        
        return selected
    except (ValueError, IndexError) as e:
        print(f"❌ Invalid selection: {e}")
        return []


# Example usage functions
def example_basic_usage():
    """Example of basic deployment checking."""
    # Initialize checker (you need to provide your ONC token)
    onc_token = "YOUR_ONC_TOKEN_HERE"
    checker = HydrophoneDeploymentChecker(onc_token, debug=True)
    
    # Get all deployments
    all_deployments = checker.get_all_hydrophone_deployments()
    checker.print_deployment_summary(all_deployments)
    
    # Find deployments for a specific time range
    deployments = checker.find_deployments_by_time_range(
        start_date="2020-01-01",
        end_date="2020-12-31",
        timezone_str="America/Vancouver"
    )
    
    # Check data availability
    available_deployments = checker.check_data_availability(
        deployments, 
        dtparse.parse("2020-01-01"), 
        dtparse.parse("2020-12-31")
    )
    
    print("\nDeployments with available data:")
    checker.print_deployment_summary(available_deployments)


def example_interactive_usage():
    """Example of interactive deployment selection."""
    # Initialize checker
    onc_token = "YOUR_ONC_TOKEN_HERE"
    checker = HydrophoneDeploymentChecker(onc_token)
    
    # Interactive selection
    selected_deployments = interactive_deployment_selector(checker)
    
    if selected_deployments:
        print(f"You selected {len(selected_deployments)} deployment(s)")
        # Now you can use these deployments for downloading data
    else:
        print("No deployments selected")


if __name__ == "__main__":
    # Run example
    print("Example usage:")
    print("1. Basic usage - example_basic_usage()")
    print("2. Interactive usage - example_interactive_usage()")
    print("\nMake sure to set your ONC_TOKEN in the examples!") 