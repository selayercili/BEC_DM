"""
Galaxy Data Manager for Astrophysical BEC Dark Matter Detection Simulation
Handles downloading, organizing, and selecting galaxy parameters from SPARC database
"""

import os
import pandas as pd
import numpy as np
import requests
import zipfile
from pathlib import Path
import yaml
from typing import Dict, List, Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GalaxyDataManager:
    """Manages galaxy data download, organization, and parameter extraction"""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data manager
        
        Args:
            data_dir: Directory to store galaxy data
        """
        self.data_dir = Path(data_dir)
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed" 
        self.galaxy_parameters_dir = self.data_dir / "galaxy_parameters"
        
        # Create directories
        for dir_path in [self.data_dir, self.raw_data_dir, 
                        self.processed_data_dir, self.galaxy_parameters_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        self.sparc_url = "http://astroweb.cwru.edu/SPARC/Rotmod_LTG.zip"
        self.galaxy_data = None
        
    def download_sparc_data(self, force_redownload: bool = False) -> bool:
        """
        Download SPARC galaxy database
        
        Args:
            force_redownload: If True, redownload even if data exists
            
        Returns:
            bool: True if successful, False otherwise
        """
        zip_path = self.raw_data_dir / "Rotmod_LTG.zip"
        extract_path = self.raw_data_dir / "SPARC"
        
        # Check if already downloaded
        if extract_path.exists() and not force_redownload:
            logger.info("SPARC data already exists. Use force_redownload=True to redownload.")
            return True
            
        try:
            logger.info("Downloading SPARC galaxy database...")
            response = requests.get(self.sparc_url, stream=True)
            response.raise_for_status()
            
            # Download with progress
            total_size = int(response.headers.get('content-length', 0))
            with open(zip_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\rDownload progress: {percent:.1f}%", end="")
            
            print()  # New line after progress
            logger.info("Download completed. Extracting files...")
            
            # Extract zip file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.raw_data_dir)
                
            logger.info("SPARC data successfully downloaded and extracted!")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading SPARC data: {e}")
            return False
    
    def load_galaxy_catalog(self) -> pd.DataFrame:
        """
        Load and parse the SPARC galaxy catalog
        
        Returns:
            DataFrame: Galaxy catalog with parameters
        """
        try:
            # First check if SPARC data exists
            sparc_dir = self.raw_data_dir / "SPARC"
            if not sparc_dir.exists():
                logger.warning("SPARC data not found. Attempting to download...")
                if not self.download_sparc_data():
                    logger.error("Failed to download SPARC data")
                    return self._create_mock_catalog()
            
            # Look for catalog files
            catalog_files = list(sparc_dir.glob("*.dat")) + list(sparc_dir.glob("*.txt"))
            
            if not catalog_files:
                logger.warning("No catalog files found. Creating mock data for testing...")
                return self._create_mock_catalog()
            
            # Try to find main catalog
            main_catalog = None
            priority_names = ['table1', 'catalog', 'sparc', 'galaxy']
            
            for priority in priority_names:
                for file in catalog_files:
                    if priority in file.name.lower():
                        main_catalog = file
                        break
                if main_catalog:
                    break
            
            if main_catalog is None:
                main_catalog = catalog_files[0]
                
            logger.info(f"Loading galaxy catalog from: {main_catalog}")
            
            # Try different parsing methods
            df = self._parse_catalog_file(main_catalog)
            
            if df.empty:
                logger.warning("Failed to parse catalog. Creating mock data...")
                return self._create_mock_catalog()
            
            self.galaxy_data = df
            logger.info(f"Loaded {len(df)} galaxies from catalog")
            return df
            
        except Exception as e:
            logger.error(f"Error loading galaxy catalog: {e}")
            logger.info("Creating mock catalog for testing...")
            return self._create_mock_catalog()
    
    def _parse_catalog_file(self, file_path: Path) -> pd.DataFrame:
        """
        Parse catalog file with multiple fallback methods
        
        Args:
            file_path: Path to catalog file
            
        Returns:
            DataFrame: Parsed catalog data
        """
        parse_methods = [
            {'sep': None, 'comment': '#', 'header': 0},
            {'sep': '\s+', 'comment': '#', 'header': 0, 'engine': 'python'},
            {'sep': '\t', 'comment': '#', 'header': 0},
            {'sep': ',', 'comment': '#', 'header': 0},
            {'sep': None, 'comment': '#', 'header': None},
        ]
        
        for method in parse_methods:
            try:
                df = pd.read_csv(file_path, **method)
                if not df.empty and len(df.columns) > 3:
                    # Basic validation - should have multiple columns
                    logger.info(f"Successfully parsed with method: {method}")
                    return self._standardize_columns(df)
            except Exception as e:
                logger.debug(f"Parse method {method} failed: {e}")
                continue
        
        return pd.DataFrame()
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names based on common SPARC format
        
        Args:
            df: Raw DataFrame
            
        Returns:
            DataFrame: DataFrame with standardized columns
        """
        # Common SPARC column mappings
        column_mapping = {
            'galaxy': 'Galaxy', 'name': 'Galaxy', 'gal': 'Galaxy',
            'type': 'Morph', 'morph': 'Morph', 'morphology': 'Morph',
            'dist': 'D', 'distance': 'D', 'd_mpc': 'D',
            'inc': 'Inc', 'incl': 'Inc', 'inclination': 'Inc',
            'lum': 'L_3.6', 'luminosity': 'L_3.6', 'l36': 'L_3.6',
            'mdisk': 'M_disk', 'm_disk': 'M_disk', 'stellar_mass': 'M_disk',
            'mgas': 'M_gas', 'm_gas': 'M_gas', 'gas_mass': 'M_gas',
            'vflat': 'V_flat', 'v_flat': 'V_flat', 'rotation_velocity': 'V_flat',
            'quality': 'Q', 'qual': 'Q', 'flag': 'Q'
        }
        
        # Apply mapping
        df.columns = df.columns.str.lower().str.strip()
        df = df.rename(columns=column_mapping)
        
        # If we don't have standard columns, create them with generic names
        if 'Galaxy' not in df.columns and len(df.columns) > 0:
            standard_names = ['Galaxy', 'Morph', 'D', 'Inc', 'L_3.6', 'M_disk', 'M_gas', 'V_flat', 'Q']
            for i, col in enumerate(df.columns[:len(standard_names)]):
                if col not in standard_names:
                    df = df.rename(columns={col: standard_names[i]})
        
        return df
    
    def _create_mock_catalog(self) -> pd.DataFrame:
        """
        Create mock galaxy catalog for testing when real data unavailable
        
        Returns:
            DataFrame: Mock galaxy catalog
        """
        logger.info("Creating mock galaxy catalog with representative parameters...")
        
        # Create realistic mock data based on typical galaxy properties
        mock_galaxies = [
            {'Galaxy': 'NGC3031', 'Morph': 'SA(s)ab', 'D': 3.63, 'Inc': 59, 'L_3.6': 2.5e10, 'M_disk': 1.8e10, 'M_gas': 3.2e9, 'V_flat': 230, 'Q': 3},
            {'Galaxy': 'NGC7793', 'Morph': 'SA(s)d', 'D': 3.91, 'Inc': 50, 'L_3.6': 1.1e9, 'M_disk': 2.8e9, 'M_gas': 1.5e9, 'V_flat': 110, 'Q': 3},
            {'Galaxy': 'DDO154', 'Morph': 'Im', 'D': 4.3, 'Inc': 66, 'L_3.6': 1.2e8, 'M_disk': 1.5e8, 'M_gas': 4.2e8, 'V_flat': 45, 'Q': 2},
            {'Galaxy': 'NGC2403', 'Morph': 'SAB(s)cd', 'D': 3.22, 'Inc': 63, 'L_3.6': 4.8e9, 'M_disk': 6.1e9, 'M_gas': 2.1e9, 'V_flat': 135, 'Q': 3},
            {'Galaxy': 'NGC3198', 'Morph': 'SB(rs)c', 'D': 13.8, 'Inc': 72, 'L_3.6': 8.9e9, 'M_disk': 1.2e10, 'M_gas': 1.8e9, 'V_flat': 150, 'Q': 3},
            {'Galaxy': 'UGC02259', 'Morph': 'Sm', 'D': 4.5, 'Inc': 35, 'L_3.6': 3.5e8, 'M_disk': 4.2e8, 'M_gas': 8.1e8, 'V_flat': 65, 'Q': 2},
            {'Galaxy': 'NGC6946', 'Morph': 'SAB(rs)cd', 'D': 5.9, 'Inc': 33, 'L_3.6': 1.4e10, 'M_disk': 1.8e10, 'M_gas': 3.5e9, 'V_flat': 180, 'Q': 3},
            {'Galaxy': 'IC2574', 'Morph': 'SAB(s)m', 'D': 4.02, 'Inc': 53, 'L_3.6': 6.8e8, 'M_disk': 8.5e8, 'M_gas': 1.9e9, 'V_flat': 75, 'Q': 2},
        ]
        
        df = pd.DataFrame(mock_galaxies)
        self.galaxy_data = df
        
        logger.info(f"Created mock catalog with {len(df)} galaxies")
        return df
    
    def select_representative_galaxies(self, n_galaxies: int = 10, 
                                     selection_criteria: Optional[Dict] = None) -> pd.DataFrame:
        """
        Select a representative subset of galaxies for simulation
        
        Args:
            n_galaxies: Number of galaxies to select
            selection_criteria: Dictionary with selection criteria
            
        Returns:
            DataFrame: Selected galaxies
        """
        if self.galaxy_data is None:
            self.load_galaxy_catalog()
            
        df = self.galaxy_data.copy()
        
        # Default selection criteria
        if selection_criteria is None:
            selection_criteria = {
                'min_distance': 1,     # Mpc, close enough for good data
                'max_distance': 50,    # Mpc, not too distant
                'min_v_flat': 50,      # km/s, substantial rotation
                'quality_min': 2       # Quality flag (if available)
            }
        
        # Apply filters
        if 'D' in df.columns:  # Distance column
            df = df[(df['D'] >= selection_criteria.get('min_distance', 0)) & 
                   (df['D'] <= selection_criteria.get('max_distance', 100))]
        
        if 'V_flat' in df.columns:  # Rotation velocity
            df = df[df['V_flat'] >= selection_criteria.get('min_v_flat', 0)]
            
        if 'Q' in df.columns:  # Quality flag
            df = df[df['Q'] >= selection_criteria.get('quality_min', 1)]
        
        # Select diverse sample
        if len(df) > n_galaxies:
            # Sort by different parameters to get diversity
            df_sorted = df.sort_values(['V_flat', 'D'], ascending=[True, True])
            # Take every nth galaxy to get good spread
            step = len(df_sorted) // n_galaxies
            selected_indices = np.arange(0, len(df_sorted), step)[:n_galaxies]
            df = df_sorted.iloc[selected_indices]
        
        logger.info(f"Selected {len(df)} representative galaxies")
        return df.reset_index(drop=True)
    
    def extract_galaxy_parameters(self, galaxy_name: str) -> Dict:
        """
        Extract detailed parameters for a specific galaxy
        
        Args:
            galaxy_name: Name of the galaxy
            
        Returns:
            Dict: Dictionary with galaxy parameters for BEC simulation
        """
        if self.galaxy_data is None:
            self.load_galaxy_catalog()
            
        # Find galaxy in catalog
        galaxy_row = self.galaxy_data[self.galaxy_data['Galaxy'] == galaxy_name]
        
        if galaxy_row.empty:
            logger.warning(f"Galaxy {galaxy_name} not found in catalog")
            return {}
        
        row = galaxy_row.iloc[0]
        
        # Safely extract parameters with proper type conversion
        def safe_get(column, default, dtype=float):
            try:
                value = row.get(column, default)
                if pd.isna(value) or value == '' or value == 'nan':
                    return default
                return dtype(value) if dtype != str else str(value)
            except (ValueError, TypeError):
                return default
        
        parameters = {
            'name': str(galaxy_name),
            'distance_Mpc': safe_get('D', 10.0),
            'inclination_deg': safe_get('Inc', 45.0),  
            'morphology': safe_get('Morph', 'Spiral', str),
            'luminosity_L36': safe_get('L_3.6', 1e10),
            'disk_mass_Msun': safe_get('M_disk', 1e10),
            'gas_mass_Msun': safe_get('M_gas', 1e9),
            'v_flat_km_s': safe_get('V_flat', 200.0),
            'quality': safe_get('Q', 3, int),
        }
        
        # Calculate derived parameters for BEC simulation
        parameters.update(self._calculate_derived_parameters(parameters))
        
        return parameters
    
    def _calculate_derived_parameters(self, params: Dict) -> Dict:
        """
        Calculate derived astrophysical parameters needed for BEC simulation
        
        Args:
            params: Basic galaxy parameters
            
        Returns:
            Dict: Derived parameters
        """
        # Physical constants
        G = 6.67430e-11  # m³ kg⁻¹ s⁻¹
        Msun = 1.989e30  # kg
        pc = 3.086e16    # m
        kpc = 1000 * pc  # m
        c = 2.998e8      # m/s
        hbar = 1.0546e-34  # J⋅s
        
        # Convert input parameters
        v_flat = params['v_flat_km_s'] * 1000  # Convert to m/s
        distance = params['distance_Mpc'] * 1e6 * pc  # Convert to meters
        M_disk = params['disk_mass_Msun'] * Msun  # Convert to kg
        
        # Estimate galaxy scale parameters
        # Typical disk scale length ~ 3 kpc for spiral galaxies
        scale_length_kpc = 3.0 if 'S' in params.get('morphology', 'S') else 1.5
        scale_length_m = scale_length_kpc * kpc
        
        # Dark matter halo parameters (NFW-like profile)
        # Virial radius estimation: R_vir ~ v_flat^2 / (10 * G * H0 * Omega_m)
        H0 = 70 * 1000 / (1e6 * pc)  # Hubble constant in SI units (s^-1)
        Omega_m = 0.31  # Matter density parameter
        R_vir = v_flat**2 / (10 * G * H0 * Omega_m)
        
        # Characteristic density (approximate)
        rho_char = 200 * Omega_m * (3 * H0**2) / (8 * np.pi * G)
        
        # Local dark matter density (typical value)
        rho_dm_local = 0.3e9 * 1.783e-30  # 0.3 GeV/cm³ in kg/m³
        
        derived = {
            # Basic derived quantities
            'virial_velocity_m_s': float(v_flat),
            'virial_radius_m': float(R_vir),
            'scale_length_m': float(scale_length_m),
            'scale_length_kpc': float(scale_length_kpc),
            
            # Dark matter parameters
            'dm_density_local_kg_m3': float(rho_dm_local),
            'dm_characteristic_density_kg_m3': float(rho_char),
            'dm_halo_mass_kg': float(4 * np.pi * rho_char * R_vir**3 / 3),
            
            # Characteristic scales for BEC physics
            'rotation_period_s': float(2 * np.pi * scale_length_m / v_flat),
            'rotation_period_years': float(2 * np.pi * scale_length_m / v_flat / (365.25 * 24 * 3600)),
            'dynamical_time_s': float(np.sqrt(scale_length_m**3 / (G * M_disk))),
            
            # Environmental parameters
            'magnetic_field_T': 1e-9,  # Typical galactic B-field ~1 μG
            'temperature_K': 2.7,      # CMB temperature baseline
            'cosmic_ray_density_kg_m3': 1e-27,  # Approximate
            
            # Gravitational and potential parameters  
            'potential_depth_J_kg': float(0.5 * v_flat**2),  # Per unit mass
            'escape_velocity_m_s': float(np.sqrt(2) * v_flat),
            'surface_gravity_m_s2': float(v_flat**2 / scale_length_m),
            
            # BEC-relevant parameters
            'coherence_length_estimate_m': float(hbar * v_flat / (1.67e-27 * v_flat**2)),  # Rough estimate
            'quantum_vortex_spacing_m': float(2 * np.pi * hbar / (1.67e-27 * v_flat)),  # Approximate
            
            # Detection sensitivity estimates
            'phase_shift_scale': float(G * rho_dm_local * scale_length_m**2 / (hbar * c)),
            'detection_time_s': 3600.0,  # 1 hour integration time
        }
        
        return derived
    
    def save_galaxy_parameters(self, galaxy_params: Dict, filename: Optional[str] = None):
        """
        Save galaxy parameters to YAML file
        
        Args:
            galaxy_params: Dictionary with galaxy parameters
            filename: Optional filename, defaults to galaxy name
        """
        if filename is None:
            filename = f"{galaxy_params['name']}_parameters.yaml"
            
        filepath = self.galaxy_parameters_dir / filename
        
        # Convert numpy types to Python types for YAML serialization
        serializable_params = {}
        for key, value in galaxy_params.items():
            if isinstance(value, (np.integer, np.floating, np.bool_)):
                serializable_params[key] = value.item()
            elif isinstance(value, np.ndarray):
                serializable_params[key] = value.tolist()
            else:
                serializable_params[key] = value
        
        with open(filepath, 'w') as f:
            yaml.dump(serializable_params, f, default_flow_style=False, indent=2)
            
        logger.info(f"Saved parameters for {galaxy_params['name']} to {filepath}")
    
    def load_galaxy_parameters(self, filename: str) -> Dict:
        """
        Load galaxy parameters from YAML file
        
        Args:
            filename: YAML filename
            
        Returns:
            Dict: Galaxy parameters
        """
        filepath = self.galaxy_parameters_dir / filename
        
        if not filepath.exists():
            logger.error(f"Parameter file {filepath} not found")
            return {}
        
        try:
            with open(filepath, 'r') as f:
                params = yaml.load(f, Loader=yaml.SafeLoader)
            return params if params is not None else {}
        except Exception as e:
            logger.error(f"Error loading parameters from {filepath}: {e}")
            return {}
    
    def list_available_galaxies(self) -> List[str]:
        """
        List all available galaxies in the catalog
        
        Returns:
            List[str]: List of galaxy names
        """
        if self.galaxy_data is None:
            self.load_galaxy_catalog()
            
        if self.galaxy_data is not None and 'Galaxy' in self.galaxy_data.columns:
            return self.galaxy_data['Galaxy'].tolist()
        else:
            return []
    
    def get_simulation_ready_galaxies(self, n_galaxies: int = 5) -> List[Dict]:
        """
        Get a list of galaxies ready for BEC simulation with all parameters
        
        Args:
            n_galaxies: Number of galaxies to prepare
            
        Returns:
            List[Dict]: List of galaxy parameter dictionaries
        """
        # Select representative galaxies
        selected_galaxies = self.select_representative_galaxies(n_galaxies)
        
        simulation_galaxies = []
        
        for _, galaxy_row in selected_galaxies.iterrows():
            galaxy_name = galaxy_row['Galaxy']
            params = self.extract_galaxy_parameters(galaxy_name)
            
            if params:  # If parameters extracted successfully
                # Save to file
                self.save_galaxy_parameters(params)
                simulation_galaxies.append(params)
                
        logger.info(f"Prepared {len(simulation_galaxies)} galaxies for simulation")
        return simulation_galaxies
    
    def print_galaxy_summary(self, galaxy_params: Dict):
        """
        Print a summary of galaxy parameters
        
        Args:
            galaxy_params: Galaxy parameter dictionary
        """
        print(f"\n=== Galaxy: {galaxy_params['name']} ===")
        print(f"Distance: {galaxy_params['distance_Mpc']:.1f} Mpc")
        print(f"Morphology: {galaxy_params['morphology']}")
        print(f"Rotation velocity: {galaxy_params['v_flat_km_s']:.1f} km/s")
        print(f"Stellar mass: {galaxy_params['disk_mass_Msun']:.2e} M☉")
        print(f"Gas mass: {galaxy_params['gas_mass_Msun']:.2e} M☉")
        print(f"DM density: {galaxy_params['dm_density_local_kg_m3']:.2e} kg/m³")
        print(f"Virial velocity: {galaxy_params['virial_velocity_m_s']:.0f} m/s")
        print(f"Magnetic field: {galaxy_params['magnetic_field_T']:.2e} T")