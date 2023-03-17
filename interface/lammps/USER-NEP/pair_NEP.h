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

#ifdef PAIR_CLASS

PairStyle(nep, PairNEP)

#else

#ifndef LMP_PAIR_NEP_H
#define LMP_PAIR_NEP_H

#include "nep.h"
#include "pair.h"
#include <string>
#include <Eigen/Dense>

namespace LAMMPS_NS
{
class PairNEP : public Pair
{
public:
  double cutoff;
  Eigen::MatrixXf z;
  Eigen::MatrixXf y_sample;
  Eigen::MatrixXf preds;
  Eigen::MatrixXf catted_vals;
  Eigen::ArrayXi row_indices;
  Eigen::ArrayXi col_indices;

  NEP3 nep_model;
  PairNEP(class LAMMPS*);
  virtual ~PairNEP();
  virtual void coeff(int, char**);
  virtual void settings(int, char**);
  virtual double init_one(int, int);
  virtual void init_style();
  virtual void compute(int, int);

  Eigen::MatrixXf forecast(const Eigen::MatrixXf &, const Eigen::VectorXf &, const int, const int);
  void setup() override;

protected:
  bool inited;
  char model_filename[1000];
  double cutoffsq;
  void allocate();

  int look_back;
  int lags;
  int eigvals_to_keep;
  double alpha;
  bool calc_true_forces = true;
  int global_forces_saved = 0;
};
} // namespace LAMMPS_NS

#endif
#endif
