# Applies shared compiler and linker flags.

function(LuminaApplyFlags target)
    if(MSVC)
        target_compile_options(${target} PRIVATE
            /O2 /Ob2 /Oi /GL /arch:AVX2 /fp:precise /DNDEBUG /MD
            /Zc:__cplusplus /permissive-
        )
        target_link_options(${target} PRIVATE
            /LTCG /STACK:8388608
        )
    else()
        target_compile_options(${target} PRIVATE
            -O3 -DNDEBUG -mavx2 -mfma -ffp-contract=on
        )
    endif()

    set_target_properties(${target} PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION TRUE
    )
endfunction()
