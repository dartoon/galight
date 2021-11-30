#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Import from Connor Bottrell for HSC image download
https://github.com/cbottrell
"""
import os,sys,time,requests
import tarfile,tempfile, getpass

def make_cutout_list(object_id,ra,dec,tmp_dir,dr='dr3',rerun='s20a_wide',
                     filters='GRIZY',fov_arcsec=30, mask='true', variance='true'):
    '''
    Generate cutout list for a single object_id,ra,dec with formatting described here:
    https://hscdata.mtk.nao.ac.jp/das_quarry/dr3/manual.html#list-to-upload
    
    Filters must be a single string of capitalized filter names (default 'GRIZY').
    fov_arcec is the full-width size of the desired cutouts.
    mask and variance images are lower-case strings ('true'/'false')
    tmp_dir is the location of the output cutout list (txt file).
    '''
    
    das_header = "#? rerun filter ra dec sw sh image mask variance type # object_id\n"
    das_list = []
    for filt in filters:
        row = f" {rerun} HSC-{filt} {ra} {dec} {fov_arcsec/2}asec {fov_arcsec/2}asec true {mask} {variance} coadd # {object_id}"
        das_list.append(row)
        
    cutout_list = f"{tmp_dir}/{object_id}_list.txt"
    with open(cutout_list, "w") as f:
        f.write(das_header)
        f.write("\n".join(das_list))
    return cutout_list


def download_cutouts(cutout_list,dr,rerun,tmp_dir):
    '''
    Download cutout file from list (txt file). 
    List formatting is described here:
    https://hscdata.mtk.nao.ac.jp/das_quarry/dr3/manual.html#list-to-upload
    
    Tested for dr3 and dr4. 
    Earlier releases use different urls for the archive.
    tmp_dir is the directory where the downloaded tar file is stored.
    '''
    if os.getenv('SSP_IDR_USR') != None or os.getenv('SSP_IDR_PWD') != None:
        SSP_IDR_USR = os.getenv('SSP_IDR_USR')
        SSP_IDR_PWD = os.getenv('SSP_IDR_PWD')
    else:
        SSP_IDR_USR = input('Your HSC SSP user name:\n')
        SSP_IDR_PWD = getpass.getpass('password:\n')        
    creds = {
        "user":SSP_IDR_USR,
        "secret":SSP_IDR_PWD
    }
    tar_downld = f'{tmp_dir}/hsc_cutouts.tar' 
    with open(cutout_list, "r") as f:
        while True:
            r = requests.post(f"https://hscdata.mtk.nao.ac.jp/das_quarry/{dr}/cgi-bin/cutout",
                                files={"list":f},
                                auth=(creds["user"],creds["secret"]),
                                stream=True)
            if r.status_code!=200:
                print(f"Failed Download for: {cutout_list}. HTTP Code: {r.status_code}. Waiting 30 seconds...")
                time.sleep(30)
            else:
                break
                
    with open(tar_downld, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                
    if os.access(cutout_list,0):os.remove(cutout_list)
        
    return tar_downld
        
                
def get_cutouts(object_id,ra,dec,out_dir,dr='dr3',rerun='s20a_wide',
                filters='GRIZY',fov_arcsec=30,mask='true',variance='true'):
    '''
    Get cutouts of size fov_arcsec [arcsec] in filters for a given ra,dec. 
    object_id is arbitrary and only used for naming.
    dr and rerun are the data release and rerun.
    filters must be a single string.
    
    All intermediate steps are performed in a temporary directory.
    Cutouts go to out_dir.
    
    Example usage:
    ##############
    
    from hsc_utils import hsc_image

    object_id,ra,dec,out_dir = '36411448540270969',30.611445,-6.475494,'Cutouts'
    
    hsc_image.get_cutouts(object_id,ra,dec,out_dir,dr='dr4',rerun='s21a_wide',filters='GRIZY',fov_arcsec=60)
    '''
    
    if not os.access(out_dir,0):
        print(f'Cutout output directory {out_dir} not found. Quitting...')
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        
        cutout_list = make_cutout_list(object_id,ra,dec,tmp_dir,dr,rerun,
                                       filters,fov_arcsec,mask,variance)
        
        tar_downld = download_cutouts(cutout_list,dr,rerun,tmp_dir)
            
        with tarfile.TarFile(tar_downld, 'r') as tarball:
            tarball.extractall(tmp_dir)
            
        if os.access(tar_downld,0): 
            os.remove(tar_downld)
            
        sub_dir = os.listdir(tmp_dir)[0]
        
        os.system(f'cp {tmp_dir}/{sub_dir}/*.fits {out_dir}')

def main():
    '''
    Example usage:
    ##############
    '''

    object_id,ra,dec='36411448540270969',30.611445,-6.475494
    out_dir='/lustre/work/connor.bottrell/RealSim_HSC/Cutouts'
    
    get_cutouts(object_id,ra,dec,out_dir,dr='dr4',rerun='s21a_wide',filters='GRIZY',fov_arcsec=120)
    
if __name__=='__main__':
    
    main()
