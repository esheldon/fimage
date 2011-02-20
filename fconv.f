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
              !xmod = ix-jx 
              xmod = ix+jx 
              ! only proceed if we are in the image. zero padding.
              if (xmod.gt.0 .and. xmod.le.nx) then

                do jy=-ny/2,ny/2
                  !ymod = iy-jy
                  ymod = iy+jy
                  if (ymod.gt.0 .and. ymod.le.ny) then
                    
                    jx2=jx*jx
                    jy2=jy*jy

                    expon=0.5*(jx2*wy + jy2*wx - 2.*jx*jy*wxy)
                    if (expon .le. 100.) then
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

      function conv_images_f4(
     &image1,image2,nx1,ny1,nx2,ny2,imageout)
     &result(flag)

        ! convolve image1 with an image2.
        !
        ! image2 should have odd number of pixels in each dimension,
        ! so the center won't move.
        ! and probably centered at (nx-1)/2, (ny-1)/2 although this
        ! may depend on your needs
        !
        ! the output image should be the same size as the first image
        ! the second image is centered on each pixel of the first image
        ! for the convolution
        implicit none

        integer*4 nx1,ny1,nx2,ny2
        real*4 image1(nx1,ny1)
        real*4 image2(nx2,ny2)
        real*4 imageout(nx1,ny1)
        
        real*4 csum
        real*4 val1, val2
        integer*4 xmod1,ymod1

        integer*4 ix1,iy1, ix2,iy2
        integer*4 flag

        flag=0

        if ((mod(nx2,2) .eq. 0) .or. (mod(ny2,2) .eq. 0)) then
          ! npix must be odd in both dimensions
          flag=2**0
          return
        endif

        ! dx,dy are both one here

        ! loop over the pixels of imageout. At each

        ! location, which also corresponds to image1, then do the
        ! convolution with image2 centered on that pixel

        do ix1=1,nx1
          do iy1=1,ny1
            
            csum=0

            ! this only makes sense for nx2,ny2 odd
            do ix2=1,nx2
              
              ! potition in image1
              xmod1 = ix1 + (ix2-nx2/2-1)

              ! only proceed if we are in the image. zero padding.
              if (xmod1.gt.0 .and. xmod1.le.nx1) then

                do iy2=1,ny2

                  ymod1 = iy1 + (iy2-ny2/2-1)

                  if (ymod1.gt.0 .and. ymod1.le.ny1) then
                    
                    val1=image1(xmod1,ymod1)
                    val2=image2(ix2,iy2)
                    csum = csum + val1*val2

                  endif ! within image in y
                enddo !loop for convolution in y

              endif ! in the image in x direction?
            enddo ! loop for convolution in x

            imageout(ix1,iy1) = csum

          enddo ! loop over y pixels
        enddo ! loop over x pixels
      return
      end




