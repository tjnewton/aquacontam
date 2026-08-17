"""Data ingestion pipelines — one module per data source.

Water quality sources (monitoring data with concentrations, detection limits,
and analyte identifiers) implement the :class:`DataSource` ABC and are listed
as ``*Source`` classes below.

Auxiliary feature sources (facility locations, demographics, contamination
sites) provide ``download_*()`` and ``load_*()`` free functions. These return
GeoDataFrames or DataFrames of geospatial features, not water quality samples.
"""

from aquacontam.data.base import DataSource
from aquacontam.data.ca_geotracker import CaGeoTrackerSource
from aquacontam.data.dod_pfas import download_dod_pfas, load_dod_pfas
from aquacontam.data.ejscreen import download_ejscreen, load_ejscreen
from aquacontam.data.epa_frs import download_frs, load_frs
from aquacontam.data.mi_mpart import MiMpartSource
from aquacontam.data.mn_mdh import MnMdhSource
from aquacontam.data.mo_dnr import MoDnrSource
from aquacontam.data.nc_deq import NcDeqSource
from aquacontam.data.nj_dep import NjDepSource
from aquacontam.data.nj_private_wells import download_nj_private_wells, load_nj_private_wells
from aquacontam.data.oh_epa import OhEpaSource
from aquacontam.data.sdwis import SDWISSource
from aquacontam.data.tri import download_tri_pfas, load_tri_pfas
from aquacontam.data.tx_tceq import TxTceqSource
from aquacontam.data.ucmr3 import UCMR3Source
from aquacontam.data.ucmr5 import UCMR5Source
from aquacontam.data.wa_doh import WaDohSource
from aquacontam.data.wqp import WqpSource

__all__ = [
    "CaGeoTrackerSource",
    "DataSource",
    "MiMpartSource",
    "MnMdhSource",
    "MoDnrSource",
    "NcDeqSource",
    "NjDepSource",
    "OhEpaSource",
    "SDWISSource",
    "TxTceqSource",
    "UCMR3Source",
    "UCMR5Source",
    "WaDohSource",
    "WqpSource",
    "download_dod_pfas",
    "download_ejscreen",
    "download_frs",
    "download_nj_private_wells",
    "download_tri_pfas",
    "load_dod_pfas",
    "load_ejscreen",
    "load_frs",
    "load_nj_private_wells",
    "load_tri_pfas",
]
