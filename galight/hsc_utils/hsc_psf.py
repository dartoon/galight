#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Import from Connor Bottrell for HSC PSF download
https://github.com/cbottrell
"""
import os,sys,time,requests
import tarfile,tempfile

def make_psf_list(object_id,ra,dec,tmp_dir,dr='dr3',rerun='s20a_wide',filters='GRIZY'):
    '''
    Generate PSF list for a single object_id,ra,dec with formatting described here:
    https://hscdata.mtk.nao.ac.jp/psf/8/manual.html#Bulk_mode
    
    object_id is only meaninful for filename conventions. It is arbitrary.
    
    Filters must be a single string of capitalized filter names (default 'GRIZY').
    tmp_dir is the location of the output cutout list (txt file).
    '''
    
    das_header = "#? ra dec rerun filter type centered # object_id\n"
    das_list = []
    for filt in filters:
        if filt in ['G','R','I','Z','Y']:
            filt = filt.lower()
        row = f" {ra} {dec} {rerun} {filt} coadd false # {object_id}"
        das_list.append(row)

    psf_list = f"{tmp_dir}/{object_id}_psflist.txt"
    with open(psf_list, "w") as f:
        f.write(das_header)
        f.write("\n".join(das_list))
    return psf_list

def download_psfs(psf_list,dr,rerun,tmp_dir):
    '''
    Download PSF files from list (txt file). 
    List formatting is described here:
    https://hscdata.mtk.nao.ac.jp/psf/8/manual.html#Bulk_mode
    
    Tested for dr3 and dr4. 
    Earlier releases use different urls for the archive.
    tmp_dir is the directory where the downloaded tar file is stored.
    '''

    creds = {
        "user":os.getenv('SSP_IDR_USR'),
        "secret":os.getenv('SSP_IDR_PWD')
    }
    tar_downld = f'{tmp_dir}/hsc_psfs.tar' 
    with open(psf_list, "r") as f:
        while True:
            r = requests.post(f"https://hscdata.mtk.nao.ac.jp/psf/8/cgi/getpsf?bulk=on",
                                files={"list":f},
                                auth=(creds["user"],creds["secret"]),
                                stream=True)
            if r.status_code!=200:
                print(f"Failed Download for: {psf_list}. HTTP Code: {r.status_code}. Waiting 30 seconds...")
                time.sleep(30)
            else:
                break
                
    with open(tar_downld, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                
    if os.access(psf_list,0):os.remove(psf_list)
        
    return tar_downld

def get_psfs(object_id,ra,dec,out_dir,dr='dr3',rerun='s20a_wide',filters='GRIZY'):
    '''
    Get cutouts of size fov_arcsec [arcsec] in filters for a given ra,dec. 
    object_id is arbitrary and only used for naming.
    dr and rerun are the data release and rerun.
    filters must be a single string.
    
    All intermediate steps are performed in a temporary directory.
    Cutouts go to out_dir.
    
    Example usage:
    ##############
    
    from hsc_utils import hsc_psf

    object_id,ra,dec,out_dir = '36411448540270969',30.611445,-6.475494,'PSFs'
    
    hsc_psf.get_psfs(object_id,ra,dec,out_dir,dr='dr4',rerun='s21a_wide',filters='GRIZY')
    '''
    
    if not os.access(out_dir,0):
        print(f'PSF output directory {out_dir} not found. Quitting...')
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        
        psf_list = make_psf_list(object_id,ra,dec,tmp_dir,dr,rerun,filters)
        
        tar_downld = download_psfs(psf_list,dr,rerun,tmp_dir)
            
        with tarfile.TarFile(tar_downld, 'r') as tarball:
            tarball.extractall(out_dir)
            
        if os.access(tar_downld,0): 
            os.remove(tar_downld)

        
def main():
    '''
    Example usage:
    ##############
    '''

    object_id,ra,dec,out_dir = '36411448540270969',35.72187,-5.31932,'PSFs'
    
    get_psfs(object_id,ra,dec,out_dir,dr='dr4',rerun='s21a_wide',filters='GRIZY')

if __name__=='__main__':
    main()