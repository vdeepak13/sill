#include <iostream>
#include <string>
#include <stdexcept>

#include <prl/distributed/chord_successor_list.hpp>

#include <chord_types.h>
#include "chord/chord_prot.h" // for CHORDPROC_GETSUCCESSOR
#include "chord/misc_utils.h" // for make_chord_node
#include "chord/rpclib.h" // for doRPC
#include <id_utils.h> // for make_chordID

namespace prl {

  namespace {
    // A callback function that receives the result of the successor query
    void successor_cb(chord_nodelistextres* res,
			chord_successor_list* obj,
			clnt_stat err) {
      using namespace std;
      if (err) {
	// typically a time-out (e.g. if Chord is not running locally)
	//cerr << "getsuccessor RPC failed: " << err << "\n";
	//exit(-1);
	obj->set_result();
	//   } else if (route->status == CHORD_RPCFAILURE) {
	//     nrequests = obj->set_result();
      } else if (res->status != CHORD_OK) {
	cerr << "getsuccessor RPC bad status: " << res->status << "\n";
	exit(-1);
      } else {
	std::vector<chord_node> result(res->resok->nlist.size());
	for(size_t i = 0; i < res->resok->nlist.size(); i++)
	  result[i] = make_chord_node(res->resok->nlist[i].n);
	obj->set_result(result);
	//cout << "Previous node: " << result.x << endl;
      }
      amain_exit = true; // this will cause the amain function to exit
      // this requires a patch to the sfslite-0.8 distribution
      // see svn/projects/libs/sfslite-0.8.tar.gz 
    }
  }

  chord_successor_list::chord_successor_list(const std::string& ip_address,
					     unsigned short port)
    : contact(*new chord_node()),
      successors(*new std::vector<chord_node>) {
    contact.r.hostname = ip_address.c_str();
    contact.r.port = port;
    contact.x = make_chordID(contact.r.hostname, contact.r.port);
    contact.vnode_num = 0; // what is this?
  }

  chord_successor_list::chord_successor_list(const chord_node& contact)
    : contact(*new chord_node(contact)),
      successors(*new std::vector<chord_node>) { }

  chord_successor_list::~chord_successor_list() {
    delete &contact;
    delete &successors;
  }

  const chord_node& chord_successor_list::my_node() {
    return contact;
  }

  const chordID& chord_successor_list::my_id() {
    return contact.x;
  }

  const std::vector<chord_node>& chord_successor_list::successor_nodes() {
    chordID n = contact.x;
    chord_nodelistextres* res = New chord_nodelistextres ();
    doRPC(contact, chord_program_1, CHORDPROC_GETSUCC_EXT, &n, res,
	  wrap(&successor_cb, res, this));
    amain();
    if (success)
      return successors;
    else
      throw std::runtime_error("getsuccext RPC failed");
  }
  
  std::vector<chordID> chord_successor_list::successor_ids() {
    successor_nodes();
    std::vector<chordID> result;
    for(size_t i = 0; i < successors.size(); i++)
      result.push_back(successors[i].x);
    return result;
  }

  void chord_successor_list::set_result(const std::vector<chord_node>& nodes) {
    successors = nodes;
    success = true;
  }

  void chord_successor_list::set_result() {
    success = false;
  }

}
