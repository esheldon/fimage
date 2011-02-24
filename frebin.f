      function rebin_f4image(image,nx,ny,rimage,rnx,rny) result(flag)

        ! the dimension of rimage should be a multiple of
        ! image
        !
        ! rimage must be already set to zero in all pixels
        implicit none

        integer*4 flag

        integer*4 nx,ny
        real*4 image(nx,ny)
        integer*4 rnx,rny
        real*4 rimage(rnx,rny)

        integer*4 i,j, ii, jj
        integer*4 fac

        flag=0

        ! the rnx,rny should be fac*nx, fac*ny with
        ! fac the same and fac >= 1.  

        fac=nx/rnx
        if (ny/rny .ne. fac) then
          flag=2**0
          return
        endif
        if (fac < 1) then
          flag=2**1
          return
        endif

        do i=1,nx
          ! index in rebinned image
          ii = (i-1)/fac + 1
          do j=1,ny
            jj = (j-1)/fac + 1

            rimage(ii,jj) = rimage(ii,jj) + image(i,j)
          enddo
        enddo
      
        return
      
      end
      

