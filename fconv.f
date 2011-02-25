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


c      function exp_gauss_range(expIxx,expIyy,gIxx,gIyy) result(rng)
c        real*4 expIxx,expIyy,gIxx,gIyy
c        real*4 exp_rng, g_rng
c        real*4 rng

c        exp_rng = 7.5*sqrt( max(expIxx,expIyy) )
c        g_rng = 4.5*sqrt( max(gIxx,gIyy) )
c        rng = max(exp_rng, g_rng)
c        return
c      end




      function conv_exp_gauss_f4(expIxx,expIxy,expIyy,expXcen,expYcen,
     &gIxx, gIxy, gIyy, nsub, image, nx, ny) result (flag)

        ! smooth an exponential with a gaussian.  The exp can have
        ! any center.  Place the results in the image.
        !
        ! For a first go, I'm going to simply *add* up at the sub-pixel
        ! level rather than more complex integration
        !
        ! maybe the best way to use this is run it at high resolution
        ! and then downsample

        implicit none

        integer*4 nx,ny
        real*4 image(nx,ny)

        real*4 expIxx,expIxy,expIyy,expXcen,expYcen
        real*4 gIxx, gIxy, gIyy
        real*4 expWx,expWy,expWxy
        real*4 gWx,gWy,gWxy
        real*4 expDet, gDet

        integer*4 ix,iy
        real*4 expExpon, gExpon

        real*4 exp_rng, g_rng, rng
        integer*4 nsub
        integer*4 nstep
        real*4 stepsize
        integer*4 intx, inty
        real*4 x,y,xexp,yexp,xm,ym
        real*4 x2,y2,xm2,ym2
        real*4 csum

        real*4 flag
        flag=0

        expDet = expIxx*expIyy - expIxy**2
        if (expDet .le. 0.) then
          flag=2**0
          return
        endif

        gDet = gIxx*gIyy - gIxy**2
        if (gDet .le. 0.) then
          flag=2**1
          return
        endif

        exp_rng = 7.5*sqrt( max(expIxx,expIyy) )
        g_rng = 4.5*sqrt( max(gIxx,gIyy) )
        rng = max(exp_rng, g_rng)

        expWx=expIxx/expDet
        expWy=expIyy/expDet
        expWxy=expIxy/expDet
        gWx=gIxx/gDet
        gWy=gIyy/gDet
        gWxy=gIxy/gDet

        ! rng is units of pixels, so 
        ! nstep is just twice rng times the number of
        ! subdivisions
        stepsize=1./nsub
        nstep = 2*int(rng)*nsub +1

        print *,"xcen: ",expXcen," ycen: ",expYcen
        print *,"g_rng: ",g_rng," exp_rng: ",exp_rng," rng: ",rng
        print *," nstep: ",nstep," stepsize: ",stepsize


        do ix=1,nx
          do iy=1,ny

            ! integrate over a range -rng,-rng in both
            ! x and y. Use stepsize 1.0/nsub

            ! the gaussian is always centered on this pixel
            ! that is the x,y here starting at -rng

            x = -rng
            x2 = x*x

            ! this is position relative to the exponential
            ! center
            xexp = ix + x
            xm = xexp - expXcen
            xm2 = xm*xm
            
            csum=0
            ! integration steps intx and inty
            do intx=1,nstep
              y = -rng
              y2 = y*y

              yexp = iy+y
              ym = yexp-expYcen
              ym2 = ym*ym

              do inty=1,nstep


                gExpon=0.5*(x2*gWy + y2*gWx - 2.*x*y*gWxy)
                expExpon = sqrt(xm2*expWy+ym2*expWx-2.*xm*ym*expWxy)

                csum = csum + exp(-gExpon)*exp(-expExpon)

                y = y+stepsize
                ym = ym+stepsize
                yexp = yexp+stepsize
                y2=y*y
                ym2=ym*ym

              enddo ! integration loop y

              x = x+stepsize
              xm = xm+stepsize
              xexp = xexp+stepsize
              x2=x*x
              xm2=xm*xm

            enddo ! integration loop x
              
            image(ix,iy) = csum

          enddo
        enddo

      return
      end

      
      function conv_images_f4(
     &image1,image2,nx1,ny1,nx2,ny2,imageout)
     &result(flag)

        ! convolve image1 with an image2.
        !
        ! this is not as flexible as, e.g., fftconvolve
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




