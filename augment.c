#include "./include/dbscan.h"
#include "./include/utils.h"
#include "./include/config.h"

int main( void )
{

   load_dataset();

   augment_dataset();

   free_dataset();

   return 0;

}