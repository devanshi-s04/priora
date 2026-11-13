================================================================================
Title: Predictions of Imminent Earth Impactors Discovered by LSST
Authors: Ian Chow, Mario Jurić, R. Lynne Jones, Kathleen Kiker, Joachim Moeyens, Peter G. Brown, Aren N. Heinze, and Jacob A. Kurlander
================================================================================
Description of Contents:
This package includes the .txt files cneos_all_params.txt, cneos_bcart_orbits.txt and cneos_colours.txt. 

The file cneos_all_params.txt contains the filtered set of 343 Earth impactors with well-defined state vectors and impact energies from the CNEOS database.
The columns are as follows:
- ObjID: Each object is identified by its UTC time of impact in the form YYYYMMDD-HHmm.
- lat: The geodetic latitude of impact, in degrees north of the equator (south is negative).
- lon: The geodetic longitude of impact, in degrees east of the prime meridian (west is negative).
- ele: The altitude of impact, in kilometers above the reference geoid.
- vx, vy, vz: The Cartesian x, y, z velocity of the object at the time of impact in the Earth-centered, Earth-fixed equatorial coordinate frame, in km/s.
- impact_time: The object’s UTC time of impact, in the form YYYYMMDD-HHmmss.
- impact_energy: The object’s impact energy, in kilotons of TNT (1 kT TNT = 4.184 * 10^12 J)
- mass: The object’s computed mass from impact energy and velocity, in kg.
- diameter: The object’s computed diameter, in meters.
- type: The object’s (randomly assumed) taxonomic class (either S- or C-type).
- H_v: The object’s absolute V-band magnitude inferred from its diameter and the median albedo for its taxonomic class.
- mjd_tdb: The object’s TDB time of impact, as a Modified Julian Date (MJD).

The file cneos_bcart_orbits.txt is the input orbit file to the Solar System survey simulator Sorcha and contains each of the 343 objects' orbital parameters 
in Solar System barycentric Cartesian coordinates at the specified epoch (their time of impact).

The file cneos_colours.txt is the input physical parameters file to Sorcha and contains each object’s absolute LSST r-band magnitude 
(converted from the V-band magnitude), phase curve parameters (assumed to be G = 0.15 for all objects here) 
and assumed photometric LSST ugrizy colours for its taxonomic class.
================================================================================
