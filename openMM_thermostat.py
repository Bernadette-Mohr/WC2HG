from openmm.app import *
from openmm import *
from openmm import unit
from sys import stdout
import openmm
print(openmm.__version__)

gro = GromacsGroFile('/media/bmohr/Backup/POSTDOC/OMM_benchmark/1AKI_solv_ions.gro')
top = GromacsTopFile('/media/bmohr/Backup/POSTDOC/OMM_benchmark/topol.top',
                     periodicBoxVectors=gro.getPeriodicBoxVectors(), includeDir='/usr/local/gromacs/share/gromacs/top')
system = top.createSystem(nonbondedMethod=PME, nonbondedCutoff=1.*unit.nanometer, constraints=HBonds)
system.addForce(AndersenThermostat(300*unit.kelvin, 1/unit.picosecond))
integrator = VerletIntegrator(0.002*unit.picoseconds)
simulation = Simulation(top.topology, system, integrator)
simulation.context.setPositions(gro.positions)
simulation.minimizeEnergy(tolerance=100, maxIterations=50000)
state = simulation.context.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True,
                                    getParameters=True, getParameterDerivatives=True, getIntegratorParameters=True,
                                    enforcePeriodicBox=False)
simulation.saveState('/media/bmohr/Backup/POSTDOC/OMM_benchmark/min_andersen.xml')
with open('/media/bmohr/Backup/POSTDOC/OMM_benchmark/min_andersen.pdb', 'w') as output:
    PDBFile.writeFile(simulation.topology, state.getPositions(), output, keepIds=True)
simulation.step(50000)
simulation.reporters.append(PDBReporter('/media/bmohr/Backup/POSTDOC/OMM_benchmark/nvt_equ_andersen.pdb', 1000))
simulation.reporters.append(StateDataReporter('/media/bmohr/Backup/POSTDOC/OMM_benchmark/nvt_equ_andersen.csv', 1000,
                                              step=True, potentialEnergy=True, temperature=True))

# Add MC barostat
system.addForce(MonteCarloBarostat(1*unit.bar, 300*unit.kelvin, 25))
simulation.context.reinitialize(preserveState=True)
simulation.reporters.append(PDBReporter('/media/bmohr/Backup/POSTDOC/OMM_benchmark/npt_equ_andersen.pdb', 1000))
simulation.reporters.append(StateDataReporter('/media/bmohr/Backup/POSTDOC/OMM_benchmark/npt_equ_andersen.csv', 1000,
                                              step=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
                                              volume=True, temperature=True, append=False))
simulation.step(50000)
simulation.reporters.append(PDBReporter('/media/bmohr/Backup/POSTDOC/OMM_benchmark/prod_andersen.pdb', 5000))
simulation.reporters.append(StateDataReporter('/media/bmohr/Backup/POSTDOC/OMM_benchmark/prod_andersen.csv', 5000,
                                              step=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
                                              volume=True, temperature=True, append=False))
simulation.step(500000)
