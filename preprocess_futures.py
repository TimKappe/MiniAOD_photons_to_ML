import ROOT
from DataFormats.FWLite import Handle, Events

import os
import numpy as np
import pandas as pd
from datetime import date


from get_preselection import get_total_preselection

from typing import List, Tuple, Union
from numpy.typing import NDArray
from mytypes import Filename, Particle, Mask


ROOT.gROOT.SetBatch(True)
# load FWLite C++ libraries
ROOT.gSystem.Load("libFWCoreFWLite.so")
ROOT.gSystem.Load("libDataFormatsFWLite.so")
ROOT.FWLiteEnabler.enable()


def get_pt(photon: Particle) -> float:
    return photon.pt()

def get_et(photon: Particle) -> float:
    return photon.et()

def get_eta(photon: Particle) -> float:
    return photon.eta()

def get_phi(photon: Particle) -> float:
    return photon.phi()

def get_r9(photon: Particle) -> float:
    return photon.full5x5_r9()

def get_HoE(photon: Particle) -> float:
    return photon.hadronicOverEm()

def get_sigma_ieie(photon: Particle) -> float:
    return photon.sigmaEtaEta()

def get_isolations(photon: Particle) -> Tuple[float, float, float, float]:
    """I_ch, I_gamma, I_n, I_track"""
    return photon.chargedHadronIso(), photon.photonIso(), photon.neutralHadronIso(), photon.trackIso()

def get_ecalIso(photon: Particle) -> float:
    return photon.ecalPFClusterIso()

def get_hcalIso(photon: Particle) -> float:
    return photon.hcalPFClusterIso()

def is_real(photon: Particle, genparticles) -> bool:
    """returns True for a real photon and False for a fake
    the photon must have pdgID 22 and be truthmatched to a genParticle"""
    # first check the pdgID of the reco
    try:
        pdgId = photon.genParticle().pdgId()
        if pdgId != 22: 
            return False  # fake
    except ReferenceError:
        return False  # fake

    matched = False
    # loop through genParticles to see if one matches
    for genparticle in genparticles:
        pdgId = genparticle.pdgId()
        if pdgId != 22: continue

        if not genparticle.isPromptFinalState(): continue
        if not genparticle.fromHardProcessFinalState(): continue

        if not np.abs(photon.pt()-genparticle.pt())/photon.pt() < 0.20: continue
        
        # build four-vector to calculate DeltaR to photon 

        photon_vector = ROOT.TLorentzVector()
        photon_vector.SetPtEtaPhiE(photon.pt(), photon.eta(), photon.phi(), photon.energy())

        genParticle_vector = ROOT.TLorentzVector()
        genParticle_vector.SetPtEtaPhiE(genparticle.pt(), genparticle.eta(), genparticle.phi(), genparticle.energy())
        
        deltaR = photon_vector.DeltaR(genParticle_vector)
        if deltaR < 0.1:
            matched = True
            break
    return matched




def did_convert_full(photon: Particle) -> bool:
    """checks if photon converted and both tracks got reconstructed"""
    if photon.conversions(): return True
    else: return False

def did_convert_oneleg(photon: Particle) -> bool:
    """checks if photon converted and only one track got reconstructed"""
    if photon.conversionsOneLeg(): return True
    else: return False

def has_conversion_tracks(photon: Particle) -> bool:
    """checks if photon has any conversion tracks (one- or two-legged)"""
    if photon.hasConversionTracks(): return True
    else: return False

def get_detector_ID(photon: Particle) -> bool:
    '''returns True for Barrel and False for Endcap'''
    return photon.superCluster().seed().seed().subdetId()==1

def pass_eveto(photon: Particle) -> bool:
    return photon.passElectronVeto()

def get_mc_truth(photon: Particle) -> int:
    try:
        pdgId = photon.genParticle().pdgId()
        return pdgId
    except ReferenceError:
        return -1

def get_bdt_run2(photon: Particle) -> float:
    # mva is range -1 to 1, I use 0 to 1
    mva = photon.userFloat("PhotonMVAEstimatorRunIIFall17v2Values")
    return (mva+1)/2

def get_bdt_run3(photon: Particle) -> float:
    # mva is range -1 to 1, I use 0 to 1
    mva = photon.userFloat("PhotonMVAEstimatorRunIIIWinter22v1Values")
    return (mva+1)/2


def get_all(photon: Particle) -> dict[str, Union[int, float, bool]]:

    try:
        pdgId = photon.genParticle().pdgId()
        real = True if pdgId == 22 else False
        true_energy = photon.genParticle().energy()
    except ReferenceError:
        real = False  # fake
        true_energy = -999.

    try:
        mc_truth = photon.genParticle().pdgId()
    except ReferenceError:
        mc_truth = -1

    return {
        'pt': photon.pt(),
        'et': photon.et(),
        'eta': photon.eta(),
        'phi': photon.phi(),
        'r9': photon.full5x5_r9(),
        'HoE': photon.hadronicOverEm(),
        'sigma_ieie': photon.sigmaEtaEta(),
        'I_ch': photon.chargedHadronIso(),
        'I_gamma': photon.photonIso(),
        'I_n': photon.neutralHadronIso(),
        'I_tr': photon.trackIso(),
        'ecalIso': photon.ecalPFClusterIso(),
        'hcalIso': photon.hcalPFClusterIso(),
        # 'real': real,  # this is determined at a later point now
        # 'mc_truth': mc_truth,  # Uncomment if mc_truth is needed despite the comment in the original code
        'bdt2': (photon.userFloat("PhotonMVAEstimatorRunIIFall17v2Values") + 1) / 2,
        'bdt3': (photon.userFloat("PhotonMVAEstimatorRunIIIWinter22v1Values") + 1) / 2,
        'detID': True if photon.superCluster().seed().seed().subdetId() == 1 else False,
        'converted': True if photon.conversions() else False,
        'convertedOneLeg': True if photon.conversionsOneLeg() else False,
        'conversion_tracks': has_conversion_tracks(photon),
        'eveto': photon.passElectronVeto(),
        "true_energy": true_energy,
        "SC_raw_energy": photon.superCluster().rawEnergy()
    }


def select_rechits(recHits, photon_seed, distance=5) -> NDArray[float]:
    """
    This function selects ECAL RecHits around the seed of the photon candidate.
    Selects a square of size 2*distance+1
    """
    seed_i_eta = photon_seed.ieta()
    seed_i_phi = photon_seed.iphi()

    rechits_array: NDArray[int] = np.zeros((2 * distance + 1, 2 * distance + 1))
    for recHit in recHits:
        # get crystal indices to see if they are close to our photon
        raw_id = recHit.detid().rawId()
        ID = ROOT.EBDetId(raw_id)

        i_eta: int = ID.ieta()
        i_phi: int = ID.iphi()

        if abs(i_phi - seed_i_phi) > distance or abs(i_eta - seed_i_eta) > distance:
            continue

        rechits_array[i_eta - seed_i_eta + distance, i_phi - seed_i_phi + distance] = recHit.energy()

    if distance % 2 == 0:
        # Calculate the sum of energies for the outer rows and columns
        row_sums = rechits_array.sum(axis=1)  # Sum of each row
        col_sums = rechits_array.sum(axis=0)  # Sum of each column

        # Determine which row and column to remove
        min_row = np.argmin([row_sums[0], row_sums[-1]])
        min_col = np.argmin([col_sums[0], col_sums[-1]])

        # Remove the row and column with the minimum energy sum
        if min_row == 0:
            rechits_array = rechits_array[1:]  # Remove first row
        else:
            rechits_array = rechits_array[:-1]  # Remove last row

        if min_col == 0:
            rechits_array = rechits_array[:, 1:]  # Remove first column
        else:
            rechits_array = rechits_array[:, :-1]  # Remove last column

    return rechits_array

def detect_mode(file: Filename) -> Tuple[str, str]:
    """decide if tag and probe is being run and if it data or MC based on the filename"""
    if 'EGamma' in file:  # Zee data
        mode = 'tagprobe'
        kind = 'data'
    elif 'DYto2L-2Jets_MLL-50' in file:  # Zee sim
        mode = 'tagprobe'
        kind = 'mc'
    else:
        mode = None  # TODO choose the default for non tag and probe later
        kind = None
    return mode, kind

def matches_trigger(triggerhandle) -> bool:
    pass

def passes_tag_sel(df: dict) -> bool:
    sel = df['pt'] > 40
    sel &= df['hasPixelSeed']  # pixelSeed
    sel &= df['chargedHadronPFPVIso'] < 20
    sel &= (df['chargedHadronPFPVIso'] / df['pt']) < 0.3
    return sel

def get_zee_mc_mask(df: dict) -> Mask:
    pass

def get_inv_mass(tag: dict, probe: dict) -> float:
    mass = np.sqrt(2*tag['pt']*probe['pt'] * (
                    np.cosh(tag['eta']-probe['eta']) 
                    - np.cos(tag['phi'] - probe['phi'])
                    )
                   )
    return mass

def tagprobe_matching(df_event: List[dict], rechits_event: list[NDArray]) -> Tuple[List[dict], List[NDArray]]:
    """returns empty list to skip events not matching the criteria (instead of using continue in an loop)"""
    if len(df_event)!=2: return [], []
    part1, part2 = df_event
    inv_mass = get_inv_mass(part1, part2)
    if not ((80 < inv_mass) & (inv_mass < 100)): return [], []
    if passes_tag_sel(part1): 
        if passes_tag_sel(part2):
            tag_idx = np.random.choice([0,1])
        else: 
            tag_idx = 0
    elif passes_tag_sel(part2):
        tag_idx = 1
    else: 
        return [], []
    probe_idx = 1 - tag_idx  # 0 if 1 or 1 if 0
    df_event[tag_idx]['tagprobe'] = 'tag'
    df_event[probe_idx]['tagprobe'] = 'probe'
    for i, df in enumerate(df_event):
        df_event[i]['pair_mass'] = inv_mass
        # set other pair quantities here
    # rechits_event = [rechits_event[probe_idx]]
    rechits_event.pop(tag_idx)  # I only want the rechits of the probe
    return df_event, rechits_event

def main(file: Filename, rechitdistance: int = 5) -> Tuple[pd.DataFrame, NDArray[NDArray[float]]]:
    """loop through all events and photons per event in a given file, read ECAL recHits and photon attributes."""
    print("INFO: opening file", file.split("/")[-1])
    print('full filename:', file)
    photonHandle, photonLabel = Handle("std::vector<pat::Photon>"), "slimmedPhotons"
    RecHitHandleEB, RecHitLabelEB = Handle("edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> >"), "reducedEgamma:reducedEBRecHits"
    RecHitHandleEE, RecHitLabelEE = Handle("edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> >"), "reducedEgamma:reducedEERecHits"
    genParticlesHandle, genParticlesLabel = Handle("std::vector<reco::GenParticle>"), "prunedGenParticles"
    rhoHandle, rhoLabel = Handle("std::double"), "fixedGridRhoAll"
    triggerHandle, triggerLabel = Handle("edm::TriggerResults"), ""
    events = Events(file)

    # lists to fill in the eventloop:
    df_list: List[dict] = []  # save data in nested list to convert to DataFrame later
    rechit_list: List[NDArray] = []  # save data in nested list to convert to DataFrame later
    mode, kind = detect_mode(file)
    for i, event in enumerate(events):
        if i == 0:
            print("\tINFO: file open sucessful, starting Event processing")
        elif i+1 % 10_000 == 0:
            print(f"\tINFO: processing event {i+1}.")
        # print("\t INFO: processing event", i)
        event.getByLabel(photonLabel, photonHandle)
        event.getByLabel(RecHitLabelEB, RecHitHandleEB)
        event.getByLabel(RecHitLabelEE, RecHitHandleEE)
        if mode!='tagprobe':
            event.getByLabel(genParticlesLabel, genParticlesHandle)
        event.getByLabel(rhoLabel, rhoHandle)
        event.getByLabel(triggerLabel, triggerHandle)


        # # ignore for now
        # if mode == 'tagprobe' and kind == 'data':
            # if not matches_trigger(triggerHandle): continue

        df_event: List[dict] = []
        rechits_event: List[NDArray] = []
        if mode != 'tagprobe':
            genParticles = genParticlesHandle.product()
        photon_number = 0
        for photon in photonHandle.product():
            # only use barrel
            if not get_detector_ID(photon): continue
            # photon_number += 1
            # print('\t\tPhoton number:', photon_number)
            
            # dataframe
            seed_id = photon.superCluster().seed().seed()
            seed_id = ROOT.EBDetId(seed_id)  # get crystal indices of photon candidate seed:

            photonAttributes = get_all(photon)
            photonAttributes["rho"] = rhoHandle.product()[0]
            photonAttributes["seed_ieta"] = seed_id.ieta()
            photonAttributes["seed_iphi"] = seed_id.iphi()
            if mode=='tagprobe':
                photonAttributes["hasPixelSeed"] = photon.hasPixelSeed()  # bool
                photonAttributes["chargedHadronPFPVIso"] = photon.chargedHadronPFPVIso() # float
            
            # add event only after preselection
            use_eveto = False if mode=='tagprobe' else True
            if not get_total_preselection(photonAttributes, use_eveto=use_eveto): continue

            # determine whether photon is real or fake
            if mode != 'tagprobe':
                photonAttributes["real"] = is_real(photon, genParticles)

            # rechits
            # using photon.EEDetId() directly gives the same value but errors in select_recHits
            # because it has no attribute ieta
            if photon.superCluster().seed().seed().subdetId() == 1:
                recHits = RecHitHandleEB.product()
            else:
                recHits = RecHitHandleEE.product()
            rechits_array = select_rechits(photon_seed=seed_id, recHits=recHits, distance=rechitdistance)

            # filter empty rechits
            if rechits_array.sum()==0: continue

            df_event += [photonAttributes]  # list of dicts with the values of the respective photon
            rechits_event += [rechits_array]
        if mode=='tagprobe':
            df_event, rechits_event = tagprobe_matching(df_event, rechits_event)
        df_list += df_event
        rechit_list += rechits_event
    print('INFO: all events processed')

    df = pd.DataFrame(df_list)  # labels are taken from the dicts in data_list
    rechits = np.array(rechit_list, dtype=np.float32)
    return df, rechits




def determine_datasite(file: Filename) -> str:
    datasite = 'T2_US_Wisconsin'  # high pt, g+jets, postEE
    if 'MGG' in file:  # mgg cut, g+jets, postEE
        datasite = 'T2_US_Caltech'  
    elif '10to40' in file:  # low pt, g+jets, postEE
        datasite = 'T1_US_FNAL_Disk'  
    elif 'EGamma' in file:  # Zee data
        datasite = 'T1_US_FNAL_Disk'  
        # datasite = 'T1_DE_KIT_Disk'  # no servers available to read the file
        # datasite = 'T1_FR_CCIN2P3_Disk'  # no servers available to read the file
    elif 'DYto2L-2Jets_MLL-50' in file:  # Zee sim
        datasite = 'T1_US_FNAL_Disk' 
    return datasite

def get_save_loc() -> str:
    """check the date and create a new directory to save the output"""
    current_date = date.today()
    formatted_date = current_date.strftime("%d%B%Y")
    savedir = f'./output{formatted_date}/'
    # in principle there is no need to distinguish between low/high pt when saving
    # they all have different names (I checked)
    if not (os.path.exists(savedir + "recHits/") and  os.path.exists(savedir + "df/")):
        os.makedirs(savedir + "df/")
        os.makedirs(savedir + "recHits/")
    return savedir
    

def process_file(file: Filename) -> None:

    datasite = determine_datasite(file)  # determine datasite from filename
    if datasite is not None:
        file = '/store/test/xrootd/' + datasite + file
    file = 'root://xrootd-cms.infn.it/' + file
    df, rechits = main(file, rechitdistance=16)

    # save stuff
    savedir = get_save_loc()
    outname: str = file.split('/')[-1].split('.')[0]  # name of input file without directory and ending
    dfname: Filename = savedir + 'df/' + outname + '.pkl'
    rechitname: str = savedir + 'recHits/' + outname + '.npy'

    df.to_pickle(dfname)
    print('INFO: photon df file saved as:', dfname)

    np.save(rechitname, rechits)
    print('INFO: recHits file saved as:', rechitname)

    print('INFO: finished running.')

if __name__ == '__main__':
    # high pt problem file:
    # process_file('/store/mc/Run3Summer22EEMiniAODv4/GJet_PT-40_DoubleEMEnriched_TuneCP5_13p6TeV_pythia8/MINIAODSIM/130X_mcRun3_2022_realistic_postEE_v6-v2/30000/2a3e6842-6a82-4c80-921a-cd7fe86dab59.root')
    # high pt test file:
    # process_file('/store/mc/Run3Summer22EEMiniAODv4/GJet_PT-40_DoubleEMEnriched_TuneCP5_13p6TeV_pythia8/MINIAODSIM/130X_mcRun3_2022_realistic_postEE_v6-v2/30000/cb93eb36-cefb-4aea-97aa-fcf8cd72245f.root')
    # mgg test file:
    #process_file('/store/mc/Run3Summer22EEMiniAODv4/GJet_PT-40_DoubleEMEnriched_MGG-80_TuneCP5_13p6TeV_pythia8/MINIAODSIM/130X_mcRun3_2022_realistic_postEE_v6-v2/50000/d9c395aa-9eee-426a-944f-9ef41058f2d3.root')
    # zee mc:
    # process_file('/store/mc/Run3Summer22EEMiniAODv4/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/MINIAODSIM/130X_mcRun3_2022_realistic_postEE_v6_ext2-v2/2820000/62dad405-af8f-4f51-ae23-b5b4619eb570.root')
    # zee data:
    process_file('/store/data/Run2022G/EGamma/MINIAOD/19Dec2023-v1/2560000/44613402-63f2-4bf0-9485-36b3ab13d45f.root')
