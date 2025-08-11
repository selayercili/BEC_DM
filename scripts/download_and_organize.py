#!/usr/bin/env python3
"""
Simple script to download and organize galaxy data for BEC dark matter simulation
"""

import sys
import os
from pathlib import Path

# Add src directory to path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from data import GalaxyDataManager
except ImportError as e:
    print(f"Error importing data module: {e}")
    print("Make sure you have created src/data.py file")
    sys.exit(1)

def main():
    print("=" * 50)
    print("Galaxy Data Download and Organization")
    print("=" * 50)
    
    # Initialize data manager
    print("\n1. Initializing data manager...")
    manager = GalaxyDataManager(data_dir="data")
    
    # Download SPARC data
    print("\n2. Downloading SPARC galaxy database...")
    success = manager.download_sparc_data()
    
    if success:
        print("✅ SPARC data downloaded successfully!")
    else:
        print("⚠️  Download failed, will use mock data for testing")
    
    # Load galaxy catalog
    print("\n3. Loading galaxy catalog...")
    catalog = manager.load_galaxy_catalog()
    
    if not catalog.empty:
        print(f"✅ Loaded {len(catalog)} galaxies")
        print(f"Columns: {list(catalog.columns)}")
    else:
        print("❌ Failed to load catalog")
        return
    
    # Select representative galaxies for simulation
    print("\n4. Selecting galaxies for simulation...")
    n_galaxies = 5  # Keep it simple - just 5 galaxies
    
    selected_galaxies = manager.select_representative_galaxies(
        n_galaxies=n_galaxies,
        selection_criteria={
            'min_distance': 1,      # At least 1 Mpc away
            'max_distance': 20,     # Not more than 20 Mpc away
            'min_v_flat': 50,       # Minimum rotation velocity
            'quality_min': 2        # Good quality data
        }
    )
    
    if not selected_galaxies.empty:
        print(f"✅ Selected {len(selected_galaxies)} galaxies:")
        for i, row in selected_galaxies.iterrows():
            print(f"  - {row['Galaxy']}: {row.get('V_flat', 'N/A')} km/s, {row.get('D', 'N/A')} Mpc")
    else:
        print("❌ Failed to select galaxies")
        return
    
    # Extract parameters and save for each galaxy
    print("\n5. Extracting parameters for each galaxy...")
    simulation_ready = []
    
    for i, row in selected_galaxies.iterrows():
        galaxy_name = row['Galaxy']
        print(f"  Processing {galaxy_name}...")
        
        # Extract all parameters
        params = manager.extract_galaxy_parameters(galaxy_name)
        
        if params:
            # Save to file
            manager.save_galaxy_parameters(params)
            simulation_ready.append(params)
            print(f"    ✅ Saved parameters for {galaxy_name}")
        else:
            print(f"    ❌ Failed to extract parameters for {galaxy_name}")
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Successfully processed {len(simulation_ready)} galaxies")
    print(f"Parameter files saved in: data/galaxy_parameters/")
    
    if simulation_ready:
        print("\nGalaxies ready for BEC simulation:")
        for params in simulation_ready:
            print(f"\n📁 {params['name']}:")
            print(f"   Distance: {params['distance_Mpc']:.1f} Mpc")
            print(f"   Rotation: {params['v_flat_km_s']:.0f} km/s") 
            print(f"   Type: {params['morphology']}")
            print(f"   DM density: {params['dm_density_local_kg_m3']:.2e} kg/m³")
            print(f"   File: {params['name']}_parameters.yaml")
    
    print(f"\n🎉 Setup complete! Ready to run BEC simulations.")
    print(f"Next step: Create your simulation scripts in the notebooks/ folder")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()