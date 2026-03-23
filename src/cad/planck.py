#!/usr/bin/env python
"""
Script to downgrade Planck maps to nside=1024, convert to μK units, rotate to Celestial coordinates, and save them
with both high and low confidence masks.
"""
import os
import healpy as hp
import numpy as np
from os.path import join as opj
from astropy.io import fits

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def change_coord(m, coord, lmax=None):
    """Change coordinates using spherical harmonic transformations."""
    rot = hp.Rotator(coord=coord)
    if isinstance(m, list):
        rotated_maps = []
        for mm in m:
            nside = hp.get_nside(mm)
            lmax = 3 * nside - 1 if lmax is None else lmax
            alms = hp.map2alm(mm, lmax=lmax)
            rot.rotate_alm(alms, inplace=True)
            rotated_maps.append(hp.alm2map(alms, nside=nside, lmax=lmax))
        return rotated_maps
    else:
        nside = hp.get_nside(m)
        lmax = 3 * nside - 1 if lmax is None else lmax
        alms = hp.map2alm(m, lmax=lmax)
        rot.rotate_alm(alms, inplace=True)
        return hp.alm2map(alms, nside=nside, lmax=lmax)
    
def process_map(data_dir):
    """Process and downgrade SMICA map."""
    print("Processing SMICA map...")
    
    # Input file paths
    base_path = '/global/cfs/cdirs/des/ajzhou/cmb/planck/cmb'
    map_path = f'{base_path}/COM_CMB_IQU-smica_2048_R3.00_full.fits'
    map_hm1_path = f'{base_path}/COM_CMB_IQU-smica_2048_R3.00_hm1.fits'
    map_hm2_path = f'{base_path}/COM_CMB_IQU-smica_2048_R3.00_hm2.fits'
    
    # Read frequency masks (already in RING ordering)
    t_mask_common = hp.read_map(f'{base_path}/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits', nest=False)
    t_mask_100 = hp.read_map(f'{base_path}/COM_Mask_Likelihood-temperature-100-hm1_2048_R3.00.fits', nest=False)
    t_mask_143 = hp.read_map(f'{base_path}/COM_Mask_Likelihood-temperature-143-hm1_2048_R3.00.fits', nest=False)
    t_mask_217 = hp.read_map(f'{base_path}/COM_Mask_Likelihood-temperature-217-hm1_2048_R3.00.fits', nest=False)

    # Extract full maps (NESTED ordering, convert K to μK)
    file = fits.open(map_path)
    t_map = file[1].data['I_STOKES'] * 1e6
    # q_map = file[1].data['Q_STOKES'] * 1e6
    # u_map = file[1].data['U_STOKES'] * 1e6
    file.close()
    
    # Load half-mission maps
    file_hm1 = fits.open(map_hm1_path)
    t_hm1 = file_hm1[1].data['I_STOKES'] * 1e6
    # q_hm1 = file_hm1[1].data['Q_STOKES'] * 1e6
    # u_hm1 = file_hm1[1].data['U_STOKES'] * 1e6
    file_hm1.close()
    
    file_hm2 = fits.open(map_hm2_path)
    t_hm2 = file_hm2[1].data['I_STOKES'] * 1e6
    # q_hm2 = file_hm2[1].data['Q_STOKES'] * 1e6
    # u_hm2 = file_hm2[1].data['U_STOKES'] * 1e6
    file_hm2.close()
    
    # Downgrade to nside=1024
    nside_out = 1024
    
    # Downgrade maps and convert to RING ordering
    t_map_down = hp.ud_grade(t_map, nside_out=nside_out, order_in='NESTED', order_out='RING')
    # q_map_down = hp.ud_grade(q_map, nside_out=nside_out, order_in='NESTED', order_out='RING')
    # u_map_down = hp.ud_grade(u_map, nside_out=nside_out, order_in='NESTED', order_out='RING')
    
    t_hm1_down = hp.ud_grade(t_hm1, nside_out=nside_out, order_in='NESTED', order_out='RING')
    # q_hm1_down = hp.ud_grade(q_hm1, nside_out=nside_out, order_in='NESTED', order_out='RING')
    # u_hm1_down = hp.ud_grade(u_hm1, nside_out=nside_out, order_in='NESTED', order_out='RING')
    
    t_hm2_down = hp.ud_grade(t_hm2, nside_out=nside_out, order_in='NESTED', order_out='RING')
    # q_hm2_down = hp.ud_grade(q_hm2, nside_out=nside_out, order_in='NESTED', order_out='RING')
    # u_hm2_down = hp.ud_grade(u_hm2, nside_out=nside_out, order_in='NESTED', order_out='RING')
    
    # Downgrade frequency masks
    t_mask_common = hp.ud_grade(t_mask_common, nside_out=nside_out, order_in='RING', order_out='RING')
    t_mask_100 = hp.ud_grade(t_mask_100, nside_out=nside_out, order_in='RING', order_out='RING')
    t_mask_143 = hp.ud_grade(t_mask_143, nside_out=nside_out, order_in='RING', order_out='RING')
    t_mask_217 = hp.ud_grade(t_mask_217, nside_out=nside_out, order_in='RING', order_out='RING')
    
    # Convert to equatorial (Celestial) coordinates
    print(f'rotating full mission maps to celestial coordinates')
    t_map_down = change_coord(t_map_down, ['G', 'C'])
    # q_map_down = change_coord(q_map_down, ['G', 'C'])
    # u_map_down = change_coord(u_map_down, ['G', 'C'])
    
    print(f'rotating half mission maps (1) to celestial coordinates')
    t_hm1_down = change_coord(t_hm1_down, ['G', 'C'])
    # q_hm1_down = change_coord(q_hm1_down, ['G', 'C'])
    # u_hm1_down = change_coord(u_hm1_down, ['G', 'C'])
    
    print(f'rotating half mission maps (2) to celestial coordinates')
    t_hm2_down = change_coord(t_hm2_down, ['G', 'C'])
    # q_hm2_down = change_coord(q_hm2_down, ['G', 'C'])
    # u_hm2_down = change_coord(u_hm2_down, ['G', 'C'])
    
    print(f'rotating masks to celestial coordinates')
    t_mask_common = change_coord(t_mask_common, ['G', 'C'])
    t_mask_100 = change_coord(t_mask_100, ['G', 'C'])
    t_mask_143 = change_coord(t_mask_143, ['G', 'C'])
    t_mask_217 = change_coord(t_mask_217, ['G', 'C'])
    
    # Ensure masks remain binary after rotation
    t_mask_common = np.where(t_mask_common > 0.5, 1.0, 0.0)
    t_mask_100 = np.where(t_mask_100 > 0.5, 1.0, 0.0)
    t_mask_143 = np.where(t_mask_143 > 0.5, 1.0, 0.0)
    t_mask_217 = np.where(t_mask_217 > 0.5, 1.0, 0.0)
    
    # Save maps
    hp.write_map(opj(data_dir, 'smica_t_map_1024.fits'), t_map_down, overwrite=True)
    # hp.write_map(opj(data_dir, 'smica_q_map_1024.fits'), q_map_down, overwrite=True)
    # hp.write_map(opj(data_dir, 'smica_u_map_1024.fits'), u_map_down, overwrite=True)
    
    hp.write_map(opj(data_dir, 'smica_t_hm1_1024.fits'), t_hm1_down, overwrite=True)
    # hp.write_map(opj(data_dir, 'smica_q_hm1_1024.fits'), q_hm1_down, overwrite=True)
    # hp.write_map(opj(data_dir, 'smica_u_hm1_1024.fits'), u_hm1_down, overwrite=True)
    
    hp.write_map(opj(data_dir, 'smica_t_hm2_1024.fits'), t_hm2_down, overwrite=True)
    # hp.write_map(opj(data_dir, 'smica_q_hm2_1024.fits'), q_hm2_down, overwrite=True)
    # hp.write_map(opj(data_dir, 'smica_u_hm2_1024.fits'), u_hm2_down, overwrite=True)
    
    # Save masks
    hp.write_map(opj(data_dir, 'smica_t_mask_common_1024.fits'), t_mask_common, overwrite=True)
    hp.write_map(opj(data_dir, 'smica_t_mask_100_1024.fits'), t_mask_100, overwrite=True)
    hp.write_map(opj(data_dir, 'smica_t_mask_143_1024.fits'), t_mask_143, overwrite=True)
    hp.write_map(opj(data_dir, 'smica_t_mask_217_1024.fits'), t_mask_217, overwrite=True)
    
    print("SMICA maps processed and saved to:", data_dir)

class PlanckMapLoader:
    """Class to load processed Planck maps at nside=1024."""
    
    def __init__(self, data_dir='/global/homes/j/junzhez/isotropy/data/cmb_full_sky'):
        """Initialize with data directory path."""
        self.data_dir = data_dir
        
    def load_smica_TQU1024(self, load_pol=False):
        """
        Load temperature (and optionally polarization) maps at nside=1024, RING, Celestial.
        Polarization I/O in process_map / file reads is commented out; load_pol should stay False until restored.
        """
        # Load temperature maps
        t_map = hp.read_map(os.path.join(self.data_dir, 'smica_t_map_1024.fits'))
        t_hm1 = hp.read_map(os.path.join(self.data_dir, 'smica_t_hm1_1024.fits'))
        t_hm2 = hp.read_map(os.path.join(self.data_dir, 'smica_t_hm2_1024.fits'))
        
        # Load masks
        t_mask_common = hp.read_map(os.path.join(self.data_dir, 'smica_t_mask_common_1024.fits'))
        t_mask_100 = hp.read_map(os.path.join(self.data_dir, 'smica_t_mask_100_1024.fits'))
        t_mask_143 = hp.read_map(os.path.join(self.data_dir, 'smica_t_mask_143_1024.fits'))
        t_mask_217 = hp.read_map(os.path.join(self.data_dir, 'smica_t_mask_217_1024.fits'))
        
        # Initialize result dictionary with temperature maps
        result = {
            'T': t_map,
            'T_hm1': t_hm1,
            'T_hm2': t_hm2,
            'T_mask_common': t_mask_common,
            'T_mask_100': t_mask_100,
            'T_mask_143': t_mask_143,
            'T_mask_217': t_mask_217
        }
        
        # if load_pol:
        #     q_map = hp.read_map(os.path.join(self.data_dir, 'smica_q_map_1024.fits'))
        #     u_map = hp.read_map(os.path.join(self.data_dir, 'smica_u_map_1024.fits'))
        #     q_hm1 = hp.read_map(os.path.join(self.data_dir, 'smica_q_hm1_1024.fits'))
        #     u_hm1 = hp.read_map(os.path.join(self.data_dir, 'smica_u_hm1_1024.fits'))
        #     q_hm2 = hp.read_map(os.path.join(self.data_dir, 'smica_q_hm2_1024.fits'))
        #     u_hm2 = hp.read_map(os.path.join(self.data_dir, 'smica_u_hm2_1024.fits'))
        #     result.update({
        #         'Q': q_map, 'U': u_map, 'Q_hm1': q_hm1, 'U_hm1': u_hm1, 'Q_hm2': q_hm2, 'U_hm2': u_hm2,
        #     })
        if load_pol:
            raise NotImplementedError("Polarization maps are disabled; re-enable in planck.process_map and loader.")
        
        return result

def main():
    # Create data directory if needed
    data_dir = '/global/homes/j/junzhez/isotropy/data/cmb_full_sky'
    ensure_dir(data_dir)
    
    # Process the smica map
    process_map(data_dir)

if __name__ == "__main__":
    main()