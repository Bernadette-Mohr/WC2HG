import sys
import re
import subprocess
import os
import mdtraj as md
import matplotlib.pyplot as plt
import numpy as np
import time
import random

import openpathsampling as paths
from openpathsampling.engines import gromacs as ops_gmx
from openpathsampling.engines.openmm.tools import ops_load_trajectory
from openpathsampling.engines.openmm.topology import MDTrajTopology

from extra_functions import *


def analyze_path(sequence):
    options = {
        'gmx_executable': 'gmx ',
        'snapshot_timestep': 20,
        'base_dir': ".",
        'prefix': "md"
    }

    # md_engine = ops_gmx.Engine(gro='./phase1-' + sequence + "/stable-" + sequence + '.gro',
    #                              mdp="res/md.mdp",
    #                              top='./phase1-' + sequence + "/stable-" + sequence + '.top',
    #                              options=options,
    #                              base_dir=".",
    #                              prefix="md").named("engine")

    md_engine = ops_gmx.Engine(gro='mdX.gro',
                               mdp="md.mdp",
                               top='topol.top',
                               options=options,
                               base_dir=".",
                               prefix="md").named("engine")

    template = md_engine.current_snapshot

    # wc = md.load('./phase1-' + sequence + "/stable-" + sequence + '.xtc',top='./phase1-' + sequence + "/stable-" + sequence + '.gro')
    wc = md.load('0.trr', top='mdX.gro')

    topology = wc.topology
    resida = len(sequence) // 2

    print('name "N9" and resid ' + str(resida))
    ahg = topology.select('name "N7" and resid ' + str(resida))[0]
    abs = topology.select('name "N6" and resid ' + str(resida))[0]
    awc = topology.select('name "N1" and resid ' + str(resida))[0]
    thg = topology.select('name "N3" and resid ' + str(len(sequence) * 2 - resida - 1))[0]
    tbs = topology.select('name "O4" and resid ' + str(len(sequence) * 2 - resida - 1))[0]
    twc = topology.select('name "N3" and resid ' + str(len(sequence) * 2 - resida - 1))[0]
    ha = topology.select('name "H3" and resid ' + str(len(sequence) * 2 - resida - 1))[0]

    d_WC = paths.MDTrajFunctionCV("d_WC", md.compute_distances, md_engine.current_snapshot.topology,
                                  atom_pairs=[[awc, twc]])
    d_HG = paths.MDTrajFunctionCV("d_HG", md.compute_distances, md_engine.current_snapshot.topology,
                                  atom_pairs=[[ahg, thg]])
    d_BS = paths.MDTrajFunctionCV("d_BS", md.compute_distances, md_engine.current_snapshot.topology,
                                  atom_pairs=[[abs, tbs]])

    a_hg = paths.MDTrajFunctionCV("a_hg", md.compute_angles, md_engine.current_snapshot.topology,
                                  angle_indices=[[ha, ahg, thg]])
    a_wc = paths.MDTrajFunctionCV("a_wc", md.compute_angles, md_engine.current_snapshot.topology,
                                  angle_indices=[[ha, awc, twc]])

    # traj = ops_load_trajectory('./phase1-' + sequence + "/stable-" + sequence + '.xtc',top='./phase1-' + sequence + "/stable-" + sequence + '.gro')
    # trajectory = paths.Trajectory([md_engine.read_frame_from_file('./phase2-' + sequence + '/000.trr', num) for num in range(310)])

    bondlist = []
    bondlist.append(topology.select('name N1 and resid 2 or name N3 and resid 7'))
    bondlist.append(topology.select('name N7 and resid 2 or name N3 and resid 7'))
    bondlist.append(topology.select('name N6 and resid 2 or name O4 and resid 7'))

    bpdistwc = md.compute_distances(wc, atom_pairs=bondlist, periodic=True)

    wclist = list()
    hglist = list()
    for f in range(wc.n_frames):
        wclist.append(bpdistwc[f, 0])
        hglist.append(bpdistwc[f, 1])

    plt.plot(wclist, label='WC distance')
    plt.plot(hglist, label="HG")
    plt.title("Plumed simulation " + sequence)
    plt.savefig("show.png")


def combine(file1, file2):
    os.system('gmx trjconv -f ' + file1 + ' -o 000.trr')
    os.system('cp ' + file2 + ' 001.xtc')

    options = {
        'gmx_executable': 'gmx ',
        'snapshot_timestep': 0.002,
        'base_dir': ".",
        'prefix': "md"
    }

    md_engine = ops_gmx.Engine(gro="md22.gro",
                               mdp="res/md2.mdp",
                               top="GTATG-topol.top",
                               options=options,
                               base_dir=".",
                               prefix="md").named("engine")

    # count = 1
    # trarr = [md_engine.read_frame_from_file("000.trr", count)]
    #
    # while trarr[-1] != None:
    #     count += 1
    #     trarr.append(md_engine.read_frame_from_file("000.trr", count))
    #
    #     if count > 20:
    #         print(trarr)
    #
    # trarr.pop(-1)
    # trajectory = paths.Trajectory(trarr)
    # print(len(trarr))
    #
    # print(trajectory.to_dict()['snapshots'])

    traj = md.load(file1, top="md22.gro")
    print(traj.time, traj.n_frames)


def run_ops(grofile="md16.gro", mdpfile="md.mdp", topfile="GTATG-topol.top", trrfile="md.trr"):
    os.system('cp ' + trrfile + ' 000.trr')
    os.system('rm initial*')
    # Engine

    options = {
        'gmx_executable': 'gmx ',
        'snapshot_timestep': 20,
        'base_dir': ".",
        'prefix': "md"
    }

    md_engine = ops_gmx.Engine(gro=grofile,
                               mdp=mdpfile,
                               top=topfile,
                               options=options,
                               base_dir=".",
                               prefix="md").named("engine")

    wc = md.load("000.trr", top=grofile)
    topology = wc.topology
    print(len(wc.xyz[0]))
    # input()

    bondlist = []
    bondlist.append(topology.select('name N1 and resid 2 or name N3 and resid 7'))
    bondlist.append(topology.select('name N7 and resid 2 or name N3 and resid 7'))
    bondlist.append(topology.select('name N6 and resid 2 or name O4 and resid 7'))

    ha = topology.select('name "H3" and resid 7')[0]

    # Collective Variable
    template = md_engine.current_snapshot
    d_WC = paths.MDTrajFunctionCV("d_WC", md.compute_distances, md_engine.current_snapshot.topology,
                                  atom_pairs=[bondlist[0]])
    d_HG = paths.MDTrajFunctionCV("d_HG", md.compute_distances, md_engine.current_snapshot.topology,
                                  atom_pairs=[bondlist[1]])
    d_BS = paths.MDTrajFunctionCV("d_BS", md.compute_distances, md_engine.current_snapshot.topology,
                                  atom_pairs=[bondlist[2]])

    a_hg = paths.MDTrajFunctionCV("a_hg", md.compute_angles, md_engine.current_snapshot.topology,
                                  angle_indices=[[ha] + bondlist[1]])
    a_wc = paths.MDTrajFunctionCV("a_wc", md.compute_angles, md_engine.current_snapshot.topology,
                                  angle_indices=[[ha] + bondlist[1]])

    # Volumes
    distarr2 = [0, 0.35]  # Hoeken weer toevoegen!

    WC = (
            paths.CVDefinedVolume(d_WC, lambda_min=distarr2[0], lambda_max=distarr2[1]) &
            paths.CVDefinedVolume(d_BS, lambda_min=distarr2[0], lambda_max=distarr2[1])
    ).named("WC")

    HG = (
            paths.CVDefinedVolume(d_HG, lambda_min=distarr2[0], lambda_max=distarr2[1]) &
            paths.CVDefinedVolume(d_BS, lambda_min=distarr2[0], lambda_max=distarr2[1])
    ).named("noWC")

    # initial trajectory
    trajectory = paths.Trajectory([md_engine.read_frame_from_file("000.trr", num)
                                   for num in range(len(wc.xyz[0]))])

    # Network
    network = paths.TPSNetwork(initial_states=HG, final_states=WC)

    print("hier")
    plt.plot(d_WC(trajectory), d_HG(trajectory))

    plt.xlabel("Hydrogen bond distance WC")
    plt.ylabel("Hydrogen bond distance HG")
    plt.title("Rotation")
    plt.savefig("deze.png")

    # Emsembles

    subtrajectories = [network.analysis_ensembles[0].split(trajectory)]
    print(subtrajectories)

    # plt.plot(d_WC(trajectory), 'k.-')
    for subtrajectory in subtrajectories[0]:
        plt.plot(d_WC(subtrajectory), d_HG(subtrajectory), )

    plt.xlabel("Hydrogen bond distance WC")
    plt.ylabel("Hydrogen bond distance HG")
    plt.title("Rotation")

    # Scheme
    scheme = paths.OneWayShootingMoveScheme(network,
                                            selector=paths.UniformSelector(),
                                            engine=md_engine)

    # Initale condition
    initial_conditions = scheme.initial_conditions_from_trajectories(subtrajectories[0][0])
    scheme.assert_initial_conditions(initial_conditions)

    # Storage
    rndmletters = List("QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm0123456789")
    fname = "".join(random.sample(rndmletters, 8))

    storage = paths.Storage(fname + ".nc", "w", md_engine.current_snapshot)

    sampler = paths.PathSampling(storage,
                                 move_scheme=scheme,
                                 sample_set=initial_conditions)

    sampler.run(20)


def HGorWC(trrfile, top="40+16/md16_3.gro", letters=['A', 'T'], resid=[2, 7], check=None):
    wc = md.load(trrfile, top=top)
    topology = wc.topology

    # # Collective Variable
    # template = md_engine.current_snapshot
    # d_WC = paths.MDTrajFunctionCV("d_WC", md.compute_distances, md_engine.current_snapshot.topology, atom_pairs=[[84, 243]])
    # d_HG = paths.MDTrajFunctionCV("d_HG", md.compute_distances, md_engine.current_snapshot.topology, atom_pairs=[[78, 243]])
    # d_BS = paths.MDTrajFunctionCV("d_BS", md.compute_distances, md_engine.current_snapshot.topology, atom_pairs=[[81, 242]])
    #
    # distarr = [0, 0.3, 0.3]
    # WC = (
    #     paths.CVDefinedVolume(d_WC, lambda_min=distarr[0], lambda_max=distarr[1]) &
    #     paths.CVDefinedVolume(d_BS, lambda_min=distarr[0], lambda_max=distarr[1])
    # ).named("WC")
    #
    # HG = (
    #     paths.CVDefinedVolume(d_HG, lambda_min=distarr[0], lambda_max=distarr[2]) &
    #     paths.CVDefinedVolume(d_BS, lambda_min=distarr[0], lambda_max=distarr[2])
    # ).named("noWC")

    # trajectory = paths.Trajectory([md_engine.read_frame_from_file(trrfile, num)
    #                     for num in range(wc.n_frames)])

    hgbonds = {'A': [('N1', 'N3'), ('N7', 'N3'), ('N6', 'O4')],
               'C': [('N3', 'N1'), ('O2', 'N2'), ('N3', 'N7'), ('N4', 'O6')],
               'G': [('N1', 'N3'), ('N2', 'O2'), ('N7', 'N3'), ('O6', 'N4')],
               'T': [('N3', 'N1'), ('N3', 'N7'), ('O4', 'N6')]}

    bondlist = []

    for atoms in hgbonds[letters[0]]:
        # print('name '+ atoms[0] +' and resid '+ str(index) + ' or name '+ atoms[1] +' and resid '+ str(len(letters)*2 -1 - index))
        bondlist.append(topology.select(
            'name ' + atoms[0] + ' and resid ' + str(resid[0]) + ' or name ' + atoms[1] + ' and resid ' + str(
                resid[1])))

    bpdistwc = md.compute_distances(wc, atom_pairs=bondlist, periodic=True)

    checkarr = [False, False]  # wc, hg
    frame = [0, 0]
    for f in range(wc.n_frames):
        if bpdistwc[f, 0] < 0.35 and bpdistwc[f, 2] < 0.35 and not checkarr[1]:
            checkarr[1] = True
            frame[1] = f
        if bpdistwc[f, 1] < 0.35 and bpdistwc[f, 2] < 0.35 and not checkarr[0]:
            checkarr[0] = True
            frame[0] = f
        if all(checkarr):
            break

    if check == None:
        print(checkarr, frame)
    elif check == 'WC' and checkarr[0]:
        print('This trajectory is first in WC at frame ' + str(frame[0]))
        return temperature
    elif check == 'HG' and checkarr[1]:
        print('This trajectory is first in HG at frame ' + str(frame[1]))
        return True
    elif check == 'both':
        print(checkarr, frame)
        return checkarr
    else:
        print('Nope!')
        return False


def pathdensity(directory='trrgoed', top="mdwater.gro", gro="mdwater.gro", mdp="md.mdp",
                storagefile='proteingoed.txt', bins=50, output='deze', short=False, minpath=3):
    """ Pathdensity """

    options = {
        'gmx_executable': 'gmx ',
        'snapshot_timestep': 20,
        'base_dir': ".",
        'prefix': "md"
    }

    residA = str(6)  # place of A
    residT = str(15)  # place of T

    neighbors = ['T', 'A', 'T', 'A']  # before A, After A, Before T, After T

    file = open(storagefile, 'a+')
    md_engine = ops_gmx.Engine(gro=gro,
                               mdp=mdp,
                               top=top,
                               options=options,
                               base_dir=".",
                               prefix="md").named("engine")

    wc = md.load(directory + '/' + os.listdir(directory)[0], top=top)
    topology = wc.topology

    ha = topology.select('name "H3" and resid ' + residT)[0]

    bondlist = []
    bondlist.append(topology.select('name N1 and resid ' + residA + ' or name N3 and resid ' + residT))
    bondlist.append(topology.select('name N7 and resid ' + residA + ' or name N3 and resid ' + residT))
    bondlist.append(topology.select('name N6 and resid ' + residA + ' or name O4 and resid ' + residT))

    anglist = []
    anglist.append(topology.select(
        'name N1 and resid ' + residA + ' or name N3 and resid ' + residT + ' or name H3 and resid ' + residT))

    dicACGT = {'G': ' and name N9 C8 H8 N7 C5 C6 O6 N1 H1 C2 N2 H21 H22 N3 C4',
               'A': ' and name N9 C8 H8 N7 C5 C6 N6 H61 H62 N1 C2 H2 N3 C4',
               'C': ' and name N1 C6 H6 C5 H5 C4 N4 H41 H42 N3 C2 O2',
               'T': ' and name N1 C6 H6 C5 C7 H71 H72 H73 C4 O4 N3 H3 C2 O2'}

    angI = topology.select('resid ' + residA + dicACGT['A'])
    angII = topology.select(
        'resid ' + residA + ' and name P OP1 OP2 or resid ' + str(int(residA) - 1) + ' and name P OP1 OP2')
    subIIIa = topology.select('resid ' + str(int(residA) - 1) + dicACGT[neighbors[0]])
    subIIIb = topology.select('resid ' + str(int(residA) + 1) + dicACGT[neighbors[1]])
    subIIIc = topology.select('resid ' + str(int(residT) - 1) + dicACGT[neighbors[2]])
    subIIId = topology.select('resid ' + str(int(residT) + 1) + dicACGT[neighbors[3]])

    angIII = np.concatenate((subIIIa, subIIIb, subIIIc, subIIId))
    print(angII)
    print(subIIIa, subIIIb, subIIIc, subIIId)

    # bpn3 = topology.select ('name N3 and resid '+residA)
    # bpn1 = topology.select ('name N1 and resid '+residA)
    # bpn7 = topology.select ('name N7 and resid '+residA)

    backbone = topology.select('resid 10 20 and name P')
    rollingbase = topology.select('resid 6 and name N7 N3 N1')

    # bb2 = topology.select ('resid 41 and name P')

    # Collective Variable
    template = md_engine.current_snapshot
    d_WC = paths.MDTrajFunctionCV("d_WC", md.compute_distances, md_engine.current_snapshot.topology,
                                  atom_pairs=[bondlist[0]])
    d_HG = paths.MDTrajFunctionCV("d_HG", md.compute_distances, md_engine.current_snapshot.topology,
                                  atom_pairs=[bondlist[1]])
    d_BS = paths.MDTrajFunctionCV("d_BS", md.compute_distances, md_engine.current_snapshot.topology,
                                  atom_pairs=[bondlist[2]])

    a_hg = paths.MDTrajFunctionCV("a_hg", md.compute_angles, md_engine.current_snapshot.topology,
                                  angle_indices=[[ha] + bondlist[1]])
    a_wc = paths.MDTrajFunctionCV("a_wc", md.compute_angles, md_engine.current_snapshot.topology,
                                  angle_indices=[[ha] + bondlist[1]])

    distarr2 = [0, 0.35]
    WC = (
            paths.CVDefinedVolume(d_WC, lambda_min=distarr2[0], lambda_max=distarr2[1]) &
            paths.CVDefinedVolume(d_BS, lambda_min=distarr2[0], lambda_max=distarr2[1])
    ).named("WC")

    HG = (
            paths.CVDefinedVolume(d_HG, lambda_min=distarr2[0], lambda_max=distarr2[1]) &
            paths.CVDefinedVolume(d_BS, lambda_min=distarr2[0], lambda_max=distarr2[1])
    ).named("noWC")

    for index, filename in enumerate(os.listdir(directory)):

        wc = md.load(directory + '/' + filename, top=top)

        # initial trajectory
        trajectory = paths.Trajectory([md_engine.read_frame_from_file(directory + '/' + filename, num)
                                       for num in range(len(wc.xyz[0]))])

        # Network
        network = paths.TPSNetwork(initial_states=HG, final_states=WC)

        print("hier", index)
        # print(trajectory[])
        bpdistwc = md.compute_distances(wc, atom_pairs=bondlist, periodic=True)
        angsetall = md.compute_angles(wc, angle_indices=anglist, periodic=True)
        proteindist = md.compute_distances(wc, [[220, 1719]],
                                           periodic=True)  # 220 is 04' resid 6, 1711 is ARG132 H11 (van eiwit D)
        minordist = md.compute_distances(wc, [[199, 485]], periodic=True)  # distance between C1'-C1'

        outputI = get_center_of_mass(wc, angI)
        outputII = get_center_of_mass(wc, angII)
        outputIII = get_center_of_mass(wc, angIII)

        # bp1 = np.subtract(bpn3, bpn1)
        # bp2 = np.subtract(bpn3, bpn7)
        # bp = np.cross(bp1, bp2)
        # bb = np.subtract(bb1, bb2)

        phia = rolling_angle(wc, backbone, rollingbase)

        lambd = list()
        distwc = list()
        disthg = list()
        hoekBS = list()
        protein = list()
        minordis = list()
        theta = list()
        phi = list()

        for f in range(wc.n_frames):
            lambd.append(np.arctan2(bpdistwc[f, 0], bpdistwc[f, 1]))
            distwc.append(bpdistwc[f, 0])
            disthg.append(bpdistwc[f, 1])
            hoekBS.append(angsetall[f, 0])
            protein.append(proteindist[f, 0])
            minordis.append(minordist[f, 0])

            vec1 = np.subtract(outputI[f], outputII[f])
            vec2 = np.subtract(outputIII[f], outputII[f])
            theta.append(angle_between_vectors(vec1, vec2))

            phi.append(phia[f, 0])

        file.write(filename + ',arctan,' + ",".join([str(el) for el in lambd]) + "\n")
        file.write(filename + ',wc,' + ",".join([str(el) for el in distwc]) + "\n")
        file.write(filename + ',hg,' + ",".join([str(el) for el in disthg]) + "\n")
        file.write(filename + ',BShoek,' + ",".join([str(el) for el in hoekBS]) + "\n")
        file.write(filename + ',protein,' + ",".join([str(el) for el in protein]) + "\n")
        file.write(filename + ',minor,' + ",".join([str(el) for el in minordis]) + "\n")
        file.write(filename + ',theta,' + ",".join([str(el) for el in theta]) + "\n")
        file.write(filename + ',phi,' + ",".join([str(el) for el in phi]) + "\n")
        # plt.plot(distwc, disthg, alpha = 0.1, c="black")

        if short and index > minpath:
            break
    #
    # plt.hist2d(distwc_all, disthg_all, bins=bins, cmap='BuPu')
    #
    #
    # plt.xlabel("Hydrogen bond distance WC")
    # plt.ylabel("Hydrogen bond distance HG")
    # plt.title("Rotation")
    # plt.savefig(output + ".png")
    file.close()


def plotpathdensity(storagefile='storage.txt', xaxis='arctan', yaxis='wc', title="Zoe's Densityploy", xlabel="Arctan",
                    ylabel="Hydrogen bond distance WC", bins=50, output='deze', short=False, minpath=3):
    file = open(storagefile, 'r')

    f1 = file.readlines()

    xarray = []
    yarray = []

    for line in f1:
        linelist = line.strip('\n').split(',')
        if linelist[1] == xaxis:
            xarray += [float(el) for el in linelist[2:]]
        if linelist[1] == yaxis:
            yarray += [float(el) for el in linelist[2:]]

    print(xarray, yarray)
    print(len(xarray), len(yarray))
    # plt.plot(xarray, yarray, alpha = 0.1, c="black")
    plt.hist2d(xarray, yarray, bins=bins, cmap='BuPu')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(output + ".png")

    file.close()
