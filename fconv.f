      ! vim: set filetype=fortran et ts=2 sw=2 sts=2 :

      function gaussconv_f4(image,nx,ny,ixx,ixy,iyy,imageout)
     &result(flag)

      ! convolve the image with a gaussian.   The scipy convolve
      ! functions do not support "ixy" oriented gaussians
      ! evaluating convolution at ix,iy
      ! thus we sum image(ix-jx, iy-jy)*gaussian(jx,jy)
      ! fill zeros outside the image boundary

        implicit none

        integer*4 nx,ny
        real*4 image(nx,ny)
        real*4 imageout(nx,ny)
        
        real*4 ixx,ixy,iyy
        real*4 det,norm,wx,wy,wxy
        real*4 csum

        integer*4 ix,iy, jx,jy
        integer*4 xmod,ymod
        integer*4 flag

        real*4 jx2,jy2
        real*4 expon
        real*4 PI 

        parameter(PI=3.141592653589793238462643383279502884197)

        flag=0

        det = ixx*iyy - ixy**2
        if (det .le. 0.) then
          flag=2**0
          return
        endif

        ! make sure the gaussian does not change the net flux
        norm = 1./(2.*PI*sqrt(det))

        wx  = ixx/det
        wxy = ixy/det
        wy  = iyy/det

        ! dx,dy are both one here
        do ix=1,nx
          do iy=1,ny

            
            csum=0
            do jx=-nx/2,nx/2
              xmod = ix-jx 
              ! only proceed if we are in the image. zero padding.
              if (xmod.gt.0 .and. xmod.le.nx) then

                do jy=-ny/2,ny/2
                  ymod = iy-jy
                  if (ymod.gt.0 .and. ymod.le.ny) then
                    
                    jx2=jx*jx
                    jy2=jy*jy

                    expon=0.5*(jx2*wy + jy2*wx - 2.*jx*jy*wxy)
                    if (expon .le. 10.8) then
                      csum = csum + image(xmod,ymod)*exp(-expon)*norm
                    endif

                  endif ! within image in y
                enddo !loop for convolution in y

              endif ! in the image in x direction?
            enddo ! loop for convolution in x

            imageout(ix,iy) = csum

          enddo ! loop over y pixels
        enddo ! loop over x pixels
      return
      end
