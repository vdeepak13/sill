#include <iostream>

#include <boost/tuple/tuple.hpp>
#include <boost/iterator/indirect_iterator.hpp>

#include <sill/base/universe.hpp>
#include <sill/factor/table_factor.hpp>
#include <sill/model/learnt_decomposable.hpp>

int main(int argc, char** argv) {

  using namespace sill;
  using namespace std;

  ///////////////////////////// VARIABLES ///////////////////////////

  // Note: the last constant in each enum is the arity of the variable
  universe u;

  // Variable: history (0)
  enum history_values { history_true,
                        history_false,
                        history_arity };
  finite_variable* history = u.new_finite_variable(history_arity);

  // Variable: CVP (1)
  enum cvp_values { cvp_low,
                    cvp_normal,
                    cvp_high,
                    cvp_arity };
  finite_variable* cvp = u.new_finite_variable(cvp_arity);

  // Variable: PCWP (2)
  enum pcwp_values { pcwp_low,
                     pcwp_normal,
                     pcwp_high,
                     pcwp_arity };
  finite_variable* pcwp = u.new_finite_variable(pcwp_arity);

  // Variable: hypovolemia (3)
  enum hypovolemia_values { hypovolemia_true,
                            hypovolemia_false,
                            hypovolemia_arity };
  finite_variable* hypovolemia = u.new_finite_variable(hypovolemia_arity);

  // Variable: LVED volume (4)
  enum lvedvolume_values { lvedvolume_low,
                           lvedvolume_normal,
                           lvedvolume_high,
                           lvedvolume_arity };
  finite_variable* lvedvolume = u.new_finite_variable(lvedvolume_arity);

  // Variable: liver failure (5)
  enum lvfailure_values { lvfailure_true,
                          lvfailure_false,
                          lvfailure_arity };
  finite_variable* lvfailure = u.new_finite_variable(lvfailure_arity);

  // Variable: stroke volume (6)
  enum strokevolume_values { strokevolume_low,
                             strokevolume_normal,
                             strokevolume_high,
                             strokevolume_arity };
  finite_variable* strokevolume = u.new_finite_variable(strokevolume_arity);

  // Variable: low output error (7)
  enum errlowoutput_values { errlowoutput_true,
                             errlowoutput_false,
                             errlowoutput_arity };
  finite_variable* errlowoutput = u.new_finite_variable(errlowoutput_arity);

  // Variable: heartrate/blood pressure (8)
  enum hrbp_values { hrbp_low,
                     hrbp_normal,
                     hrbp_high,
                     hrbp_arity };
  finite_variable* hrbp = u.new_finite_variable(hrbp_arity);

  // Variable: heartrate/EKG (9)
  enum hrekg_values { hrekg_low,
                      hrekg_normal,
                      hrekg_high,
                      hrekg_arity };
  finite_variable* hrekg = u.new_finite_variable(hrekg_arity);

  // Variable: error in cauterization (10)
  enum errcauter_values { errcauter_true,
                          errcauter_false,
                          errcauter_arity };
  finite_variable* errcauter = u.new_finite_variable(errcauter_arity);

  // Variable: heart rate ??? (11)
  enum hrsat_values { hrsat_low,
                      hrsat_normal,
                      hrsat_high,
                      hrsat_arity };
  finite_variable* hrsat = u.new_finite_variable(hrsat_arity);

  // Variable: insufficient anesthesia (12)
  enum insuffanesth_values { insuffanesth_true,
                             insuffanesth_false,
                             insuffanesth_arity  };
  finite_variable* insuffanesth = u.new_finite_variable(insuffanesth_arity);

  // Variable: anaphylaxis (13)
  enum anaphylaxis_values { anaphylaxis_true,
                            anaphylaxis_false,
                            anaphylaxis_arity };
  finite_variable* anaphylaxis = u.new_finite_variable(anaphylaxis_arity);

  // Variable: TPR (14)
  enum tpr_values { tpr_low,
                      tpr_normal,
                      tpr_high,
                      tpr_arity };
  finite_variable* tpr = u.new_finite_variable(tpr_arity);

  // Variable: expelled CO2 (15)
  enum expco2_values { expco2_zero,
                       expco2_low,
                       expco2_normal,
                       expco2_high,
                       expco2_arity };
  finite_variable* expco2 = u.new_finite_variable(expco2_arity);

  // Variable: kinked tube (16)
  enum kinkedtube_values { kinkedtube_true,
                           kinkedtube_false,
                           kinkedtube_arity };
  finite_variable* kinkedtube = u.new_finite_variable(kinkedtube_arity);

  // Variable: minimum volume (17)
  enum minvol_values { minvol_zero,
                       minvol_low,
                       minvol_normal,
                       minvol_high,
                       minvol_arity };
  finite_variable* minvol = u.new_finite_variable(minvol_arity);

  // Variable: FiO2 (18)
  enum fio2_values { fio2_low,
                       fio2_normal,
                       fio2_arity };
  finite_variable* fio2 = u.new_finite_variable(fio2_arity);

  // Variable: PVSAT (19)
  enum pvsat_values { pvsat_low,
                      pvsat_normal,
                      pvsat_high,
                      pvsat_arity };
  finite_variable* pvsat = u.new_finite_variable(pvsat_arity);

  // Variable: SAO2 (20)
  enum sao2_values { sao2_low,
                     sao2_normal,
                     sao2_high,
                     sao2_arity };
  finite_variable* sao2 = u.new_finite_variable(sao2_arity);

  // Variable: PAP (21)
  enum pap_values { pap_low,
                    pap_normal,
                    pap_high,
                    pap_arity };
  finite_variable* pap = u.new_finite_variable(pap_arity);

  // Variable: pulmembolus (22)
  enum pulmembolus_values { pulmembolus_true,
                            pulmembolus_false,
                            pulmembolus_arity };
  finite_variable* pulmembolus = u.new_finite_variable(pulmembolus_arity);

  // Variable: shunt (23)
  enum shunt_values { shunt_normal,
                      shunt_high,
                      shunt_arity };
  finite_variable* shunt = u.new_finite_variable(shunt_arity);

  // Variable: intubation (24)
  enum intubation_values { intubation_normal,
                           intubation_esophageal,
                           intubation_onesided,
                           intubation_arity };
  finite_variable* intubation = u.new_finite_variable(intubation_arity);

  // Variable: press (25)
  enum press_values { press_zero,
                      press_low,
                      press_normal,
                      press_high,
                      press_arity };
  finite_variable* press = u.new_finite_variable(press_arity);

  // Variable: disconnect (26)
  enum disconnect_values { disconnect_true,
                           disconnect_false,
                           disconnect_arity };
  finite_variable* disconnect = u.new_finite_variable(disconnect_arity);

  // Variable: minvolset (27)
  enum minvolset_values { minvolset_low,
                          minvolset_normal,
                          minvolset_high,
                          minvolset_arity };
  finite_variable* minvolset = u.new_finite_variable(minvolset_arity);

  // Variable: ventilation machine (28)
  enum ventmach_values { ventmach_zero,
                         ventmach_low,
                         ventmach_normal,
                         ventmach_high,
                         ventmach_arity };
  finite_variable* ventmach = u.new_finite_variable(ventmach_arity);

  // Variable: ventilation tube (29)
  enum venttube_values { venttube_zero,
                         venttube_low,
                         venttube_normal,
                         venttube_high,
                         venttube_arity };
  finite_variable* venttube = u.new_finite_variable(venttube_arity);

  // Variable: ventilation lung (30)
  enum ventlung_values { ventlung_zero,
                         ventlung_low,
                         ventlung_normal,
                         ventlung_high,
                         ventlung_arity };
  finite_variable* ventlung = u.new_finite_variable(ventlung_arity);

  // Variable: ventilation alv (31)
  enum ventalv_values { ventalv_zero,
                        ventalv_low,
                        ventalv_normal,
                        ventalv_high,
                        ventalv_arity };
  finite_variable* ventalv = u.new_finite_variable(ventalv_arity);

  // Variable: arterial CO2 (32)
  enum artco2_values { artco2_low,
                       artco2_normal,
                       artco2_high,
                       artco2_arity };
  finite_variable* artco2 = u.new_finite_variable(artco2_arity);

  // Variable: catechol (33)
  enum catechol_values { catechol_normal,
                         catechol_high,
                         catechol_arity };
  finite_variable* catechol = u.new_finite_variable(catechol_arity);

  // Variable: heart rate (34)
  enum hr_values { hr_low,
                   hr_normal,
                   hr_high,
                   hr_arity };
  finite_variable* hr = u.new_finite_variable(hr_arity);

  // Variable: carbon monoxide (35)
  enum co_values { co_low,
                   co_normal,
                   co_high,
                   co_arity };
  finite_variable* co = u.new_finite_variable(co_arity);

  // Variable: blood pressure (36)
  enum bp_values { bp_low,
                   bp_normal,
                   bp_high,
                   bp_arity };
  finite_variable* bp = u.new_finite_variable(bp_arity);

  /////////////////////////////// FACTORS /////////////////////////////

  std::vector<table_factor> factors;

  // P(HISTORY | LVFAILURE)
  factors.push_back(table_factor(make_domain(lvfailure, history), 1.0));
  // P(PCWP | LVEDVOLUME)
  factors.push_back(table_factor(make_domain(lvedvolume, pcwp), 1.0));
  // P(HYPOVOLEMIA)
  factors.push_back(table_factor(make_domain(hypovolemia), 1.0));
  // P(LVEDVOLUME | HYPOVOLEMIA, LVFAILURE)
  factors.push_back(table_factor(make_domain(hypovolemia, lvfailure, lvedvolume), 1.0));
  // P(LVFAILURE)
  factors.push_back(table_factor(make_domain(lvfailure), 1.0));
  // P(STROKEVOLUME | HYPOVOLEMIA, LVFAILURE)
  factors.push_back(table_factor(make_domain(hypovolemia, lvfailure, strokevolume), 1.0));
  // P(ERRLOWOUTPUT)
  factors.push_back(table_factor(make_domain(errlowoutput), 1.0));
  // P(HRBP | ERRLOWOUTPUT, HR)
  factors.push_back(table_factor(make_domain(errlowoutput, hr, hrbp), 1.0));

  table_factor cvp_factor(make_domain(lvedvolume, cvp), 1.0);
  finite_domain lvedvolume_domain(make_domain(lvedvolume));

  ////////////////////////// DECOMPOSABLE MODEL ///////////////////////

  learnt_decomposable<table_factor> model;

  model *= factors;

  learnt_decomposable<table_factor>::vertex
    cvp_connect(model.find_clique_cover(lvedvolume_domain));

  cout << "Original model:\n" << model << endl;

  cout << "Adding another clique:" << endl;
  learnt_decomposable<table_factor>::vertex
    cvp_vertex(model.add_clique(cvp_factor.arguments(), cvp_factor));
  cout << model << endl;

  cout << "Connecting clique to vertex " << cvp_connect << endl;
  model.add_edge(cvp_vertex, cvp_connect);
  cout << model << endl;

  return EXIT_SUCCESS;
}
