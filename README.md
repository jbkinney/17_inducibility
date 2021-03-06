# Data and analysis scripts for:
## Forcier et al. *Measuring cis-regulatory energetics in living cells using allelic manifolds.* bioRxiv doi:10.1101/380972 (2018) 

Directory structure:
* ``code/``: Python analysis scripts and Jupyter notebooks
* ``data/``: Metadata, raw data, and processed data for this study
  * ``constructs/``: DNA sequence constructs used in this study
  * ``literature/``: Data extracted from the prior literature
  * ``plate_reader/``: Raw data from plate reader used for Miller assays
  * ``glycerol_stocks.xlsx``: List of glycerol stocks used in this study
* ``figures/``: Computationally rendered components of the figures in the manuscript.
* ``manuscript/``: Copy of Precision dissection of cis-regulatory energetics in living cells.
* ``preprocessing/``: Code and intermediate files for processing raw data
  * ``data/``: Processed data and metadata for this study
    * ``glycerol_stocks.xlsx``: List of glycerol stocks used in this study
    * ``library_clusters.py``: CRP-RNAP spacing library clustering by glycerol stock
    * ``plate_panel.pkl``: Pandas panel of raw Miller assay data, organized by plate
    * ``seq_spacing.xlsx``: Sequence and CRP-RNAP spacing information for glycerol stocks
  * ``intermediates/``: Fits and resamplings of model to processed data
  * ``results.xlsx``: Complete summary of processed data
  * ``scripts/``: Python analysis scripts for preprocessing
    * ``18.10.30_init_fits.py``: Script for generating the fit of all data for c61, occlusion, and conjoined libraries
    * ``18.10.30_param_exp_c61.py``: Script for exploring the effects of initial parameter selection on fitted values for c61 library
    * ``18.10.30_param_exp_conj.py``: Script for exploring the effects of initial parameter selection on fitted values for global fit to all libraries
    * ``18.10.30_param_exp_occlusion.py``: Script for exploring the effects of initial parameter selection on fitted values for occlusion library
    * ``18.10.30_resamp_beta.py``: Script for resampling all libraries for \beta' values
    * ``18.10.30_resamp_c61.py``: Script for resampling the library with CRP at -61.5
    * ``18.10.30_resamp_cAMP_dilution.py``: Script for resampling the data taken with varying cAMP concentrations for c71 and occlusion libraries
    * ``18.10.30_resamp_conj.py``: Script for resampling the global fit to all libraries
    * ``18.10.30_resamp_occlusion.py``: Script for resampling the library of constructs with CRP binding occluding the RNAP binding site
    * ``18.10.30_summary_gen.py``: Script for consolidating all intermediate processed data into summaries and results.xlsx
    * ``plate_processor.py``: Script for processing raw data and generating plate_panel.pkl
* ``protocols/``: Primary protocols used in this study
