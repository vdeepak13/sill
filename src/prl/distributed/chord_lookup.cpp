#include <iostream>

#include <prl/distributed/chord_lookup.hpp>

#include <chord_types.h>
#include "chord/chord_prot.h" // for chord_findarg 
#include "chord/misc_utils.h" // for make_chord_node
#include "chord/rpclib.h" // for doRPC
#include <id_utils.h> // for make_chordID

namespace prl {

  namespace {

    // A callback function that receives the result of the findroute query
    void findroute_cb(ptr<chord_findarg> fa,
		      chord_nodelistres* route, 
		      chord_lookup* obj,
		      clnt_stat err) {
      using namespace std;
      size_t nrequests;
      // extract the last node from the route
      if (err) {
	// typically a time-out (e.g. if Chord is not running locally)
	//cerr << "findroute RPC failed: " << err << "\n";
	//exit(-1);
	nrequests = obj->set_result();
      } else if (route->status == CHORD_RPCFAILURE) {
	nrequests = obj->set_result();
      } else if (route->status != CHORD_OK) {
	cerr << "findroute RPC bad status: " << route->status << "\n";
	exit(-1);
      } else if (route->resok->nlist.size () < 1) {
	cerr << "findroute RPC returned no route!\n";
	exit(-1);
      } else {
	size_t i = route->resok->nlist.size() - 1;
	chord_node result = make_chord_node(route->resok->nlist[i]);
	nrequests = obj->set_result(fa->x, result);
      }
      delete route;
      if (nrequests == 0) {
	amain_exit = true; // this will cause the amain function to exit
	// this requires a patch to the sfslite-0.8 distribution
	// see svn/projects/libs/sfslite-0.8.tar.gz 
      }
    }
  }
  
  chord_lookup::chord_lookup(const std::string& ip_address, unsigned short port)
    : contact_node(*new chord_node),
      successor(*new std::map<chordID, chord_node>),
      ids(*new std::vector<chordID>),
      nrequests() {
    contact_node.r.hostname = ip_address.c_str();
    contact_node.r.port = port;
    contact_node.x = make_chordID(contact_node.r.hostname, contact_node.r.port);
    contact_node.vnode_num = 0; // what is this?
  }

  chord_lookup::~chord_lookup() {
    delete &contact_node;
    delete &successor;
    delete &ids;
  }

  void chord_lookup::lookup(const chordID& key) {
    ids.push_back(key);
  }

  void chord_lookup::run() {
    for(size_t i = 0; i < ids.size(); i++) {
      // issue the RPC
      ptr<chord_findarg> fa = New refcounted<chord_findarg>();
      fa->x = ids[i];
      fa->return_succs = false;
      chord_nodelistres* route = New chord_nodelistres ();
      doRPC(contact_node, chord_program_1, CHORDPROC_FINDROUTE, fa, route,
	    wrap(&findroute_cb, fa, route, this));
      nrequests++;
      
      // listen for answers in batches of 1000
      if (nrequests == 1000 || i == ids.size()-1) {
	amain(); // amain() will decrease nrequests to 0
      }
    }
  }

  size_t chord_lookup::set_result() {
    return --nrequests;
  }

  size_t chord_lookup::set_result(const chordID& key, const chord_node& node) {
    successor[key] = node;
    return --nrequests;
  }
  
  const chord_node& chord_lookup::result(const chordID& key) const {
    std::map<chordID, chord_node>::const_iterator it = successor.find(key);
    assert(it != successor.end());
    return it->second;
  }

  std::string chord_lookup::result_address(const chordID& key) const {
    return result(key).r.hostname.cstr();
  }

  bool chord_lookup::contains(const chordID& key) const {
    return successor.count(key) != 0;
  }

}
