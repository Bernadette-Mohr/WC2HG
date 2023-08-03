# This script was generated by OpenMM-Setup on 2023-07-25.

from openmm import *
from openmm import app
from openmm import unit
from openmmplumed import PlumedForce

import mdtraj as md

# Input Files

pdb = app.PDBFile('/media/bmohr/Backup/POSTDOC/WCHG/MTD/DNAWC2MAT/DNAWC2MAT-processed.pdb')
forcefield = app.ForceField('amber/protein.ff14SB.xml', 'amber/DNA.bsc1.xml', 'amber/tip3p_standard.xml')

# System Configuration

nonbondedMethod = app.PME
nonbondedCutoff = 1.1 * unit.nanometers
ewaldErrorTolerance = 0.00001
constraints = app.HBonds
rigidWater = True
constraintTolerance = 0.0001

# Integration Options

dt = 0.002 * unit.picoseconds
temperature = 300 * unit.kelvin
friction = 1.0 / unit.picosecond
pressure = 0.987 * unit.atmospheres
barostatInterval = 25

# Simulation Options

steps = 10000000
equilibrationSteps = 50000
platform = Platform.getPlatformByName('CUDA')
platformProperties = {'Precision': 'mixed'}
dcdReporter = app.DCDReporter('/media/bmohr/Backup/POSTDOC/WCHG/MTD/DNAWC2MAT/mtd_DNAWC2MAT.dcd', 1000)
dataReporter = app.StateDataReporter('/media/bmohr/Backup/POSTDOC/WCHG/MTD/DNAWC2MAT/mtd_DNAWC2MAT_data.csv', 1000,
                                     totalSteps=steps, step=True, time=True, speed=True, progress=True,
                                     elapsedTime=True, remainingTime=True, potentialEnergy=True, kineticEnergy=True,
                                     totalEnergy=True, temperature=True, volume=True, density=True, separator=';')
checkpointReporter = app.CheckpointReporter('/media/bmohr/Backup/POSTDOC/WCHG/MTD/DNAWC2MAT/mtd_DNAWC2MAT.chk', 1000)
trajReporter = md.reporters.HDF5Reporter('/media/bmohr/Backup/POSTDOC/WCHG/MTD/DNAWC2MAT/mtd_DNAWC2MAT.h5', 1000,
                                         coordinates=True, time=True, cell=True, potentialEnergy=True,
                                         kineticEnergy=True, temperature=True, velocities=True, enforcePeriodicBox=True)

# Prepare the Simulation

print('Building system...')
topology = pdb.topology
positions = pdb.positions
system = forcefield.createSystem(topology, nonbondedMethod=nonbondedMethod, nonbondedCutoff=nonbondedCutoff,
                                 constraints=constraints, rigidWater=rigidWater,
                                 ewaldErrorTolerance=ewaldErrorTolerance)
system.addForce(MonteCarloBarostat(pressure, temperature, barostatInterval))

integrator = LangevinMiddleIntegrator(temperature, friction, dt)
integrator.setConstraintTolerance(constraintTolerance)
simulation = app.Simulation(topology, system, integrator, platform, platformProperties)
simulation.context.setPositions(positions)

# Write XML serialized objects

with open("/media/bmohr/Backup/POSTDOC/WCHG/MTD/DNAWC2MAT/mtd_DNAWC2MAT_setup.xml", mode="w") as file:
    file.write(XmlSerializer.serialize(system))
with open("/media/bmohr/Backup/POSTDOC/WCHG/MTD/DNAWC2MAT/mtd_DNAWC2MAT_integrator.xml", mode="w") as file:
    file.write(XmlSerializer.serialize(integrator))

# Minimize and Equilibrate

print('Performing energy minimization...')
simulation.minimizeEnergy()
print('Equilibrating...')
simulation.context.setVelocitiesToTemperature(temperature)
simulation.step(equilibrationSteps)

# Simulate

# not sure about the setting for GRIDs...
script = """
cv2: TORSION ATOMS=215,202,200,199
metad: METAD ARG=cv2 SIGMA=0.35 HEIGHT=0.05 PACE=500 GRID_MIN=-pi GRID_MAX=pi FILE=HILLS_DNAWC2MAT
PRINT ARG=* STRIDE=500 FILE=colvar_DNAWC2MAT.out"""
system.addForce(PlumedForce(script))
simulation.context.reinitialize(preserveState=True)

print('Simulating...')
simulation.reporters.append(dcdReporter)
simulation.reporters.append(dataReporter)
simulation.reporters.append(checkpointReporter)
simulation.reporters.append(trajReporter)
simulation.currentStep = 0
simulation.step(steps)

# Not mentioned in the API, but mdtraj OpenMM HDF5Reporter needs to be explicitly closed!
trajReporter.close()

# Write file with final simulation state

simulation.saveState("/media/bmohr/Backup/POSTDOC/WCHG/MTD/DNAWC2MAT/mtd_DNAWC2MAT_final_state.xml")
