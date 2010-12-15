#include <iostream>
#include <string>
#include <stdexcept>

#include <prl/distributed/chord_predecessor.hpp>

#include <chord_types.h>
#include "chord/chord_prot.h" // for CHORDPROC_GETPREDECESSOR
#include "chord/misc_utils.h" // for make_chord_node
#include "chord/rpclib.h" // for doRPC
#include <id_utils.h> // for make_chordID

namespace prl {

  namespace {
    // A callback function that receives the result of the predecessor query
    void predecessor_cb(chord_noderes* pred,
			chord_predecessor* obj,
			clnt_stat err) {
      using namespace std;
      if (err) {
	// typically a time-out (e.g. if Chord is not running locally)
	//cerr << "getpredecessor RPC failed: " << err << "\n";
	//exit(-1);
	obj->set_result();
	//   } else if (route->status == CHORD_RPCFAILURE) {
	//     nrequests = obj->set_result();
      } else if (pred->status != CHORD_OK) {
	cerr << "getpredecessor RPC bad status: " << pred->status << "\n";
	exit(-1);
      } else {
	chord_node result = make_chord_node(*pred->resok);
	obj->set_result(result);
	//cout << "Previous node: " << result.x << endl;
      }
      amain_exit = true; // this will cause the amain function to exit
      // this requires a patch to the sfslite-0.8 distribution
      // see svn/projects/libs/sfslite-0.8.tar.gz 
    }
  }

  chord_predecessor::chord_predecessor(const std::string& ip_address,
				       unsigned short port)
    : contact(*new chord_node()),
      predecessor(*new chord_node()) {
    contact.r.hostname = ip_address.c_str();
    contact.r.port = port;
    contact.x = make_chordID(contact.r.hostname, contact.r.port);
    contact.vnode_num = 0; // what is this?
  }

  chord_predecessor::chord_predecessor(const chord_node& contact)
    : contact(*new chord_node(contact)),
      predecessor(* new chord_node()) { }

  chord_predecessor::~chord_predecessor() {
    delete &contact;
    delete &predecessor;
  }

  const chord_node& chord_predecessor::my_node() {
    return contact;
  }

  const chordID& chord_predecessor::my_id() {
    return contact.x;
  }

  const chord_node& chord_predecessor::predecessor_node() {
    chordID n = contact.x;
    chord_noderes* res = New chord_noderes ();
    doRPC(contact, chord_program_1, CHORDPROC_GETPREDECESSOR, &n, res,
	  wrap(&predecessor_cb, res, this));
    amain();
    if (success)
      return predecessor;
    else
      throw std::runtime_error("getpredecessor RPC failed");
  }
  
  const chordID& chord_predecessor::predecessor_id() {
    return predecessor_node().x;
  }

  void chord_predecessor::set_result(const chord_node& node) {
    predecessor = node;
    success = true;
  }

  void chord_predecessor::set_result() {
    success = false;
  }

}
