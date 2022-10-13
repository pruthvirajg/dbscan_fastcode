#include "./include/dbscan.h"
#include "./include/utils.h"
#include "./include/config.h"

int main( void )
{
   int clusters;

   load_dataset();

   clusters = dbscan( );

   // emit classes
   emit_classes(clusters);

   // Emit outliers (NOISE)
   emit_outliers();

   free_dataset();

   return 0;

}