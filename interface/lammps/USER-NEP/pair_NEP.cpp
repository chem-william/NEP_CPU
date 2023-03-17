/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov
   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.
   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Junjie Wang (Nanjing University)
------------------------------------------------------------------------- */

#include "pair_NEP.h"
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "nep.h"
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#define EIGEN_USE_MKL_ALL
#include <Eigen/Dense>

#define LAMMPS_VERSION_NUMBER 20220324 // use the new neighbor list starting from this version

using namespace LAMMPS_NS;
using namespace Eigen;

PairNEP::PairNEP(LAMMPS* lmp) : Pair(lmp)
{
#if LAMMPS_VERSION_NUMBER >= 20201130
  centroidstressflag = CENTROID_AVAIL;
#else
  centroidstressflag = 2;
#endif

  restartinfo = 0;
  manybody_flag = 1;

  single_enable = 0;

  inited = false;
  allocated = 0;
}

PairNEP::~PairNEP()
{
  if (copymode)
    return;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

void PairNEP::allocate()
{
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = 1; j <= n; j++)
      setflag[i][j] = 1;

  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");

  int nmax = atom->nmax;
  catted_vals = MatrixXf::Zero(look_back, nmax*3);

  allocated = 1;
}

void PairNEP::coeff(int narg, char** arg)
{
  if (!allocated)
    allocate();
}

void PairNEP::settings(int narg, char** arg)
{
  if (narg != 5)
    error->all(FLERR, "Illegal pair_style command");

  strcpy(model_filename, arg[0]);
  look_back = utils::numeric(FLERR, arg[1], false, lmp);
  lags = utils::numeric(FLERR, arg[2], false, lmp);
  eigvals_to_keep = utils::numeric(FLERR, arg[3], false, lmp);
  alpha = utils::numeric(FLERR, arg[4], false, lmp);
}

void PairNEP::init_style()
{
#if LAMMPS_VERSION_NUMBER >= 20220324
  neighbor->add_request(this, NeighConst::REQ_FULL);
#else
  int irequest = neighbor->request(this, instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
#endif

  bool is_rank_0 = (comm->me == 0);
  nep_model.init_from_file(model_filename, is_rank_0);
  inited = true;
  cutoff = nep_model.paramb.rc_radial;
  cutoffsq = cutoff * cutoff;
  int n = atom->ntypes;
  for (int i = 1; i <= n; i++)
    for (int j = 1; j <= n; j++)
      cutsq[i][j] = cutoffsq;
}

double PairNEP::init_one(int i, int j) { return cutoff; }

MatrixXf PairNEP::forecast(const MatrixXf &y,
                           const VectorXf &coefs,
                           const int look_back,
                           const int lags) {
  // Get the number of variables in the system
  const int neqs = y.cols();

  MatrixXf y_forecasts = MatrixXf::Zero(look_back * 2, neqs);
  y_forecasts.topRows(look_back) = y;

  for (int h = 0; h < look_back; ++h) {
    VectorXf f = VectorXf::Zero(neqs);
    for (int i = 1; i <= lags; ++i) {
      VectorXf prior_y = y_forecasts.row(look_back + h - i);

      for (int j = 0; j < neqs; ++j) {
        int idx = j * lags * neqs + (i - 1) * neqs;
        f[j] += coefs.segment(idx, neqs).dot(prior_y);
      }
    }

    // Update y with the forecast for the next step
    y_forecasts.row(look_back + h) = f;
  }
  return y_forecasts.bottomRows(look_back);
}

void PairNEP::setup() {
  int inum = list->inum;
  catted_vals.conservativeResize(look_back, inum*3);

  // instead of having to transpose and reverse, we precompute the indices once
  row_indices = ArrayXi::LinSpaced(lags, lags - 1, 0);
  col_indices = ArrayXi::LinSpaced(inum*3, 0, inum*3);

  int lb_minus_lags = look_back - lags;
  z = MatrixXf::Zero(lb_minus_lags, inum * 3 * lags);
}

void PairNEP::compute(int eflag, int vflag)
{
  if (eflag || vflag) {
    ev_setup(eflag, vflag);
  }
  double total_potential = 0.0;
  double total_virial[6] = {0.0};
  double* per_atom_potential = nullptr;
  double** per_atom_virial = nullptr;
  if (eflag_atom) {
    per_atom_potential = eatom;
  }
  if (cvflag_atom) {
    per_atom_virial = cvatom;
  }

  int inum = list->inum;
  int *ilist = list->ilist;
  int *NN = list->numneigh;
  int **NL = list->firstneigh;
  double **pos = atom->x;
  double **f = atom->f;
  int j = 0;
  int i = 0;
  if (calc_true_forces) {
    nep_model.compute_for_lammps(
      inum, ilist, list->numneigh, list->firstneigh, atom->type, atom->x,
      total_potential, total_virial, per_atom_potential, f, per_atom_virial);

    for (int ii = 0; ii < inum; ++ii) {
      j = ilist[ii];

      catted_vals(global_forces_saved, j + inum*0) = f[j][0];
      catted_vals(global_forces_saved, j + inum*1) = f[j][1];
      catted_vals(global_forces_saved, j + inum*2) = f[j][2];
    }
  } else {
    for (int ii = 0; ii < inum; ++ii) {
      j = ilist[ii];

      f[j][0] = preds(global_forces_saved, j + inum*0);
      f[j][1] = preds(global_forces_saved, j + inum*1);
      f[j][2] = preds(global_forces_saved, j + inum*2);
    }

    for (int i1 = 0; i1 < NN[j]; ++i1) {
      int n2 = NL[j][i1];
      double r12[3] = {
        pos[n2][0] - pos[j][0],
        pos[n2][1] - pos[j][1],
        pos[n2][2] - pos[j][2]
      };

      total_virial[0] -= r12[0] * f[j][0]; // xx
      total_virial[1] -= r12[1] * f[j][1]; // yy
      total_virial[2] -= r12[2] * f[j][2]; // zz
      total_virial[3] -= r12[0] * f[j][1]; // xy
      total_virial[4] -= r12[0] * f[j][2]; // xz
      total_virial[5] -= r12[1] * f[j][2]; // yz
    }
  }
  global_forces_saved += 1;

  if (global_forces_saved == look_back) {
    calc_true_forces = !calc_true_forces;
    if (!calc_true_forces) {
      if (inum > 0) {
        // nobs = look_back
        for (int t = lags; t < look_back; ++t) {
          int t_minus_lags = t - lags;
          z.row(t_minus_lags) = catted_vals
            .middleRows(t_minus_lags, lags)
            .leftCols(inum * 3)(row_indices, col_indices)
            .reshaped<RowMajor>()
            .transpose()
            .eval();
        }
        MatrixXf y_sample = catted_vals.bottomRows(look_back - lags).leftCols(inum*3);
    
        BDCSVD<MatrixXf> svd = z.bdcSvd(ComputeThinU | ComputeThinV);
        VectorXf s = svd.singularValues()(seq(0, eigvals_to_keep - 1));
        // VectorXd s = svd.singularValues();
        // int r = s.rows();
        // double alpha = 1e-7;
        MatrixXf D = s.cwiseQuotient((s.array().square() + alpha).matrix()).asDiagonal();
        MatrixXf params = svd.matrixV().leftCols(eigvals_to_keep) * D * svd.matrixU().transpose().topRows(eigvals_to_keep) * y_sample;
        VectorXf coefs = params.reshaped<ColMajor>().transpose().eval();
        preds = PairNEP::forecast(catted_vals, coefs, look_back, lags);
      }
    }
    global_forces_saved = 0;
  }

  if (eflag) {
    eng_vdwl += total_potential;
  }
  if (vflag) {
    for (int component = 0; component < 6; ++component) {
      virial[component] += total_virial[component];
    }
  }
}
