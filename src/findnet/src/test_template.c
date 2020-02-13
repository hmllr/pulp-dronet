#include "network.h"
#include "rt/rt_api.h"



// on fabric controller
int main () {
  	network_setup();
  	network_run_FabricController();
}
