from openmm.app import *
from openmm import *
from openmm import unit
# from sys import stdout
import openmm
print(openmm.__version__)

# Setting up the system
gro = GromacsGroFile('/media/bmohr/Backup/POSTDOC/OMM_benchmark/1AKI_solv_ions.gro')
top = GromacsTopFile('/media/bmohr/Backup/POSTDOC/OMM_benchmark/topol.top',
                     periodicBoxVectors=gro.getPeriodicBoxVectors(), includeDir='/usr/local/gromacs/share/gromacs/top')
system = top.createSystem(nonbondedMethod=PME, nonbondedCutoff=1.*unit.nanometer, constraints=HBonds)
integrator = LangevinMiddleIntegrator(300*unit.kelvin,
                                      1/unit.picosecond,
                                      0.002*unit.picoseconds)
simulation = Simulation(top.topology, system, integrator)
simulation.context.setPositions(gro.positions)

# Energy minimization
simulation.minimizeEnergy(tolerance=100, maxIterations=50000)
state = simulation.context.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True,
                                    getParameters=True, getParameterDerivatives=True, getIntegratorParameters=True,
                                    enforcePeriodicBox=False)
simulation.saveState('/media/bmohr/Backup/POSTDOC/OMM_benchmark/min_langevin.xml')
with open('/media/bmohr/Backup/POSTDOC/OMM_benchmark/min_langevin.pdb', 'w') as output:
    PDBFile.writeFile(simulation.topology, state.getPositions(), output, keepIds=True)

# NVT equilibration
simulation.reporters.append(PDBReporter('/media/bmohr/Backup/POSTDOC/OMM_benchmark/nvt_equ_langevin.pdb', 1000,
                                        enforcePeriodicBox=False))
simulation.reporters.append(StateDataReporter('/media/bmohr/Backup/POSTDOC/OMM_benchmark/nvt_equ_langevin.csv', 1000,
                                              step=True, potentialEnergy=True, temperature=True))
simulation.step(50000)

# Add MC barostat
system.addForce(MonteCarloBarostat(1*unit.bar, 300*unit.kelvin, 25))
simulation.context.reinitialize(preserveState=True)
# NPT equilibration
simulation.reporters.append(PDBReporter('/media/bmohr/Backup/POSTDOC/OMM_benchmark/npt_equ_langevin.pdb', 1000,
                                        enforcePeriodicBox=False))
simulation.reporters.append(StateDataReporter('/media/bmohr/Backup/POSTDOC/OMM_benchmark/npt_equ_langevin.csv', 1000,
                                              step=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
                                              volume=True, temperature=True, append=False))
simulation.step(50000)

# production run
simulation.reporters.append(PDBReporter('/media/bmohr/Backup/POSTDOC/OMM_benchmark/prod_langevin.pdb', 2500,
                                        enforcePeriodicBox=False))
simulation.reporters.append(StateDataReporter('/media/bmohr/Backup/POSTDOC/OMM_benchmark/prod_langevin.csv', 2500,
                                              step=True, potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
                                              volume=True, temperature=True, append=False))
simulation.step(500000)
