# simulations.py
from bec_physics import GalaxyBECSimulator
from dm_models import DM_MODELS
from data import GalaxyDataManager

# Load galaxy data
manager = GalaxyDataManager()
galaxy_params = manager.load_galaxy_parameters("NGC3031_parameters.yaml")  # Example

# Initialize simulator
simulator = GalaxyBECSimulator(galaxy_params)

# Test all DM models
for name, model in DM_MODELS.items():
    results = simulator.simulate_dm_interaction(
        model['mass'], 
        model['cross_section']
    )
    print(f"{name}: SNR = {results['snr']:.1f}, {'DETECTED' if results['detectable'] else 'not detected'}")